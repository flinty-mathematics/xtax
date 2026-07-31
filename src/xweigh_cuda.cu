#include "xweigh_cuda.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "stop_signal.hpp"

namespace {

constexpr int k_block_threads = 256;
constexpr int k_warp_size = 32;
constexpr int k_warps_per_block = k_block_threads / k_warp_size;
constexpr int k_sign_flip = 0;
constexpr int k_support_switch = 1;
constexpr int k_double_sign_flip = 2;

static_assert(sizeof(long long) == sizeof(int64_t));
static_assert(sizeof(unsigned long long) == sizeof(uint64_t));

[[noreturn]] void throw_cuda_error(cudaError_t error, const char* operation) {
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorName(error) << " ("
            << cudaGetErrorString(error) << ')';
    throw std::runtime_error(message.str());
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw_cuda_error(error, operation);
}

size_t checked_add(size_t left, size_t right, const char* description) {
    if (right > std::numeric_limits<size_t>::max() - left)
        throw std::runtime_error(std::string(description) + " size overflow");
    return left + right;
}

size_t checked_multiply(size_t left, size_t right, const char* description) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        throw std::runtime_error(std::string(description) + " size overflow");
    return left * right;
}

size_t align_up(size_t value, size_t alignment) {
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : checked_add(
        value, alignment - remainder, "shared-memory alignment");
}

struct SharedLayout {
    size_t entries_offset = 0;
    size_t gram_offset = 0;
    size_t residual_offset = 0;
    size_t minority_offset = 0;
    size_t bytes = 0;
};

SharedLayout make_shared_layout(int order, int minority_count) {
    const size_t n = static_cast<size_t>(order);
    const size_t matrix_items = checked_multiply(n, n, "matrix");
    const size_t minority_items = checked_multiply(
        n, static_cast<size_t>(minority_count), "minority");

    SharedLayout layout;
    size_t offset = checked_multiply(matrix_items, sizeof(int8_t), "entries");
    layout.gram_offset = align_up(offset, alignof(int16_t));
    offset = checked_add(
        layout.gram_offset,
        checked_multiply(matrix_items, sizeof(int16_t), "Gram"),
        "shared memory");
    layout.residual_offset = align_up(offset, alignof(long long));
    offset = checked_add(
        layout.residual_offset,
        checked_multiply(n, sizeof(long long), "row residual"),
        "shared memory");
    layout.minority_offset = align_up(offset, alignof(uint16_t));
    layout.bytes = checked_add(
        layout.minority_offset,
        checked_multiply(minority_items, sizeof(uint16_t), "minority"),
        "shared memory");
    return layout;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ == 0) return;
        const size_t bytes = checked_multiply(count_, sizeof(T), "device buffer");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ == nullptr) return;
        const cudaError_t error = cudaFree(data_);
        if (error != cudaSuccess) {
            std::cerr << "Warning: cudaFree failed: "
                      << cudaGetErrorString(error) << '\n';
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return count_; }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

template <typename T>
void copy_to_device(DeviceBuffer<T>& destination,
                    const std::vector<T>& source,
                    const char* operation) {
    if (destination.size() != source.size())
        throw std::runtime_error(std::string(operation) + ": buffer size mismatch");
    if (source.empty()) return;
    check_cuda(
        cudaMemcpy(destination.data(), source.data(),
                   checked_multiply(source.size(), sizeof(T), operation),
                   cudaMemcpyHostToDevice),
        operation);
}

template <typename T>
void copy_from_device(std::vector<T>& destination,
                      const DeviceBuffer<T>& source,
                      const char* operation) {
    if (destination.size() != source.size())
        throw std::runtime_error(std::string(operation) + ": buffer size mismatch");
    if (destination.empty()) return;
    check_cuda(
        cudaMemcpy(destination.data(), source.data(),
                   checked_multiply(destination.size(), sizeof(T), operation),
                   cudaMemcpyDeviceToHost),
        operation);
}

struct DeviceMove {
    int kind;
    int row1;
    int row2;
    int col1;
    int col2;
    int minority_pos1;
    int minority_pos2;
    int value1;
    int value2;
};

struct DeviceEvaluation {
    long long score_delta;
    long long conflict_delta;
    long long squared_delta;
    long long parity_delta;
};

struct KernelParams {
    int order;
    int weight;
    int minority_count;
    int sparse_side;
    int moves_per_launch;
    int candidate_samples;
    int target_samples;
    int moves_per_cool;
    int stuck_threshold;
    double sign_fraction;
    double double_sign_fraction;
    double switch_sign_fraction;
    double squared_objective_fraction;
    double parity_objective_fraction;
    double greedy_fraction;
    double target_fraction;
    double minimum_temperature;
    double cooling;
    double reheat;
};

struct DeviceStorage {
    int8_t* current_entries;
    int16_t* current_gram;
    long long* current_residual;
    uint16_t* current_minority;
    long long* current_score;
    long long* current_conflicts;
    int8_t* best_entries;
    long long* best_score;
    long long* best_conflicts;
    double* temperature;
    double* reheat_temperature;
    int* moves_since_improvement;
    int* cooling_counter;
    curandStatePhilox4_32_10_t* rng;
    unsigned long long* moves;
    unsigned long long* candidate_evaluations;
    int* solved;
};

__device__ size_t device_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

__device__ bool device_better_pair(long long score_a, long long conflicts_a,
                                   long long score_b, long long conflicts_b) {
    return score_a < score_b ||
           (score_a == score_b && conflicts_a < conflicts_b);
}

__device__ bool device_better_evaluation(
    const DeviceEvaluation& first,
    const DeviceEvaluation& second,
    int objective_kind) {
    const long long first_primary =
        objective_kind == 1 ? first.squared_delta
        : objective_kind == 2 ? first.parity_delta
                              : first.score_delta;
    const long long second_primary =
        objective_kind == 1 ? second.squared_delta
        : objective_kind == 2 ? second.parity_delta
                              : second.score_delta;
    if (first_primary != second_primary)
        return first_primary < second_primary;
    if (first.score_delta != second.score_delta)
        return first.score_delta < second.score_delta;
    return first.conflict_delta < second.conflict_delta;
}

__device__ int device_abs(int value) {
    return value < 0 ? -value : value;
}

__device__ long long device_residual_delta(int old_value, int new_value) {
    return static_cast<long long>(device_abs(new_value)) -
           static_cast<long long>(device_abs(old_value));
}

__device__ long long device_conflict_delta(int old_value, int new_value) {
    if (old_value == 0 && new_value != 0) return 1;
    if (old_value != 0 && new_value == 0) return -1;
    return 0;
}

__device__ long long device_squared_delta(int old_value, int new_value) {
    return static_cast<long long>(new_value) * new_value -
           static_cast<long long>(old_value) * old_value;
}

__device__ long long device_parity_delta(int old_value, int new_value) {
    return (device_abs(new_value) & 1) -
           (device_abs(old_value) & 1);
}

__device__ uint32_t random_u32(curandStatePhilox4_32_10_t& rng) {
    return curand(&rng);
}

__device__ int random_index(curandStatePhilox4_32_10_t& rng, int count) {
    const uint32_t unsigned_count = static_cast<uint32_t>(count);
    const uint32_t threshold = static_cast<uint32_t>(-unsigned_count) %
                               unsigned_count;
    uint32_t value = 0;
    do {
        value = random_u32(rng);
    } while (value < threshold);
    return static_cast<int>(value % unsigned_count);
}

__device__ double random_unit(curandStatePhilox4_32_10_t& rng) {
    return static_cast<double>(random_u32(rng)) *
           (1.0 / 4294967296.0);
}

__device__ int hot_row(curandStatePhilox4_32_10_t& rng,
                       const long long* row_residual,
                       const KernelParams& params) {
    int best = random_index(rng, params.order);
    for (int sample = 1; sample < params.target_samples; ++sample) {
        const int candidate = random_index(rng, params.order);
        if (row_residual[candidate] > row_residual[best]) best = candidate;
    }
    return best;
}

__device__ bool propose_sign_flip(curandStatePhilox4_32_10_t& rng,
                                  const int8_t* entries,
                                  const uint16_t* minority,
                                  const KernelParams& params,
                                  int row,
                                  DeviceMove& move) {
    int col = 0;
    if (params.sparse_side != 0 && params.minority_count > 0) {
        const int pos = random_index(rng, params.minority_count);
        col = static_cast<int>(
            minority[static_cast<size_t>(row) * params.minority_count + pos]);
    } else {
        const int start = random_index(rng, params.order);
        col = start;
        while (entries[static_cast<size_t>(col) * params.order + row] == 0) {
            if (++col == params.order) col = 0;
            if (col == start) return false;
        }
    }
    move.kind = k_sign_flip;
    move.row1 = row;
    move.col1 = col;
    return true;
}

__device__ bool propose_support_switch(curandStatePhilox4_32_10_t& rng,
                                       const int8_t* entries,
                                       const uint16_t* minority,
                                       const KernelParams& params,
                                       int row1,
                                       DeviceMove& move) {
    if (params.minority_count == 0 || params.order < 2) return false;
    for (int attempt = 0; attempt < 24; ++attempt) {
        int row2 = random_index(rng, params.order - 1);
        if (row2 >= row1) ++row2;
        const int pos1 = random_index(rng, params.minority_count);
        const int pos2 = random_index(rng, params.minority_count);
        int col1 = 0;
        int col2 = 0;
        if (params.sparse_side != 0) {
            col1 = static_cast<int>(
                minority[static_cast<size_t>(row1) * params.minority_count +
                         pos1]);
            col2 = static_cast<int>(
                minority[static_cast<size_t>(row2) * params.minority_count +
                         pos2]);
        } else {
            col2 = static_cast<int>(
                minority[static_cast<size_t>(row1) * params.minority_count +
                         pos1]);
            col1 = static_cast<int>(
                minority[static_cast<size_t>(row2) * params.minority_count +
                         pos2]);
        }

        const int8_t row1_cross =
            entries[static_cast<size_t>(col2) * params.order + row1];
        const int8_t row2_cross =
            entries[static_cast<size_t>(col1) * params.order + row2];
        if (col1 == col2 || row1_cross != 0 || row2_cross != 0) continue;
        if (entries[static_cast<size_t>(col1) * params.order + row1] == 0 ||
            entries[static_cast<size_t>(col2) * params.order + row2] == 0) {
            continue;
        }

        move.kind = k_support_switch;
        move.row1 = row1;
        move.row2 = row2;
        move.col1 = col1;
        move.col2 = col2;
        move.minority_pos1 = pos1;
        move.minority_pos2 = pos2;
        const int value1 = static_cast<int>(
            entries[static_cast<size_t>(col1) * params.order + row1]);
        const int value2 = static_cast<int>(
            entries[static_cast<size_t>(col2) * params.order + row2]);
        move.value1 =
            random_unit(rng) < params.switch_sign_fraction
                ? -value1 : value1;
        move.value2 =
            random_unit(rng) < params.switch_sign_fraction
                ? -value2 : value2;
        return true;
    }
    return false;
}

__device__ bool propose_double_sign_flip(
    curandStatePhilox4_32_10_t& rng,
    const int8_t* entries,
    const uint16_t* minority,
    const KernelParams& params,
    int row1,
    DeviceMove& move) {
    DeviceMove first{};
    if (!propose_sign_flip(
            rng, entries, minority, params, row1, first)) {
        return false;
    }
    for (int attempt = 0; attempt < 16; ++attempt) {
        const int row2 = random_unit(rng) < 0.5
            ? row1 : random_index(rng, params.order);
        DeviceMove second{};
        if (!propose_sign_flip(
                rng, entries, minority, params, row2, second)) {
            continue;
        }
        if (first.row1 == second.row1 &&
            first.col1 == second.col1) {
            continue;
        }
        move.kind = k_double_sign_flip;
        move.row1 = first.row1;
        move.col1 = first.col1;
        move.row2 = second.row1;
        move.col2 = second.col1;
        return true;
    }
    return false;
}

__device__ bool propose_move(curandStatePhilox4_32_10_t& rng,
                             const int8_t* entries,
                             const long long* row_residual,
                             const uint16_t* minority,
                             const KernelParams& params,
                             DeviceMove& move) {
    const bool targeted = random_unit(rng) < params.target_fraction;
    const int row = targeted
        ? hot_row(rng, row_residual, params)
        : random_index(rng, params.order);
    if (random_unit(rng) < params.double_sign_fraction &&
        propose_double_sign_flip(
            rng, entries, minority, params, row, move)) {
        return true;
    }
    if (params.weight == params.order ||
        random_unit(rng) < params.sign_fraction) {
        return propose_sign_flip(rng, entries, minority, params, row, move);
    }
    if (propose_support_switch(
            rng, entries, minority, params, row, move)) {
        return true;
    }
    return propose_sign_flip(rng, entries, minority, params, row, move);
}

__device__ DeviceEvaluation evaluate_move_warp(const DeviceMove& move,
                                                const int8_t* entries,
                                                const int16_t* gram,
                                                const KernelParams& params,
                                                int lane) {
    long long score_delta = 0;
    long long conflict_delta = 0;
    long long squared_delta = 0;
    long long parity_delta = 0;
    if (move.kind == k_sign_flip) {
        const int row = move.row1;
        const int col = move.col1;
        const int change = -2 * static_cast<int>(
            entries[static_cast<size_t>(col) * params.order + row]);
        for (int other = lane; other < params.order; other += k_warp_size) {
            if (other == row) continue;
            const int old_value =
                gram[static_cast<size_t>(row) * params.order + other];
            const int new_value = old_value + change * static_cast<int>(
                entries[static_cast<size_t>(col) * params.order + other]);
            score_delta += device_residual_delta(old_value, new_value);
            conflict_delta += device_conflict_delta(old_value, new_value);
            squared_delta += device_squared_delta(old_value, new_value);
            parity_delta += device_parity_delta(old_value, new_value);
        }
    } else if (move.kind == k_support_switch) {
        const int row1 = move.row1;
        const int row2 = move.row2;
        const int col1 = move.col1;
        const int col2 = move.col2;
        const int old_value1 = static_cast<int>(
            entries[static_cast<size_t>(col1) * params.order + row1]);
        const int old_value2 = static_cast<int>(
            entries[static_cast<size_t>(col2) * params.order + row2]);
        for (int other = lane; other < params.order; other += k_warp_size) {
            if (other == row1 || other == row2) continue;
            const int old1 =
                gram[static_cast<size_t>(row1) * params.order + other];
            const int old2 =
                gram[static_cast<size_t>(row2) * params.order + other];
            const int new1 = old1 +
                move.value1 * static_cast<int>(
                    entries[static_cast<size_t>(col2) * params.order + other]) -
                old_value1 * static_cast<int>(
                    entries[static_cast<size_t>(col1) * params.order + other]);
            const int new2 = old2 +
                move.value2 * static_cast<int>(
                    entries[static_cast<size_t>(col1) * params.order + other]) -
                old_value2 * static_cast<int>(
                    entries[static_cast<size_t>(col2) * params.order + other]);
            score_delta += device_residual_delta(old1, new1);
            score_delta += device_residual_delta(old2, new2);
            conflict_delta += device_conflict_delta(old1, new1);
            conflict_delta += device_conflict_delta(old2, new2);
            squared_delta += device_squared_delta(old1, new1);
            squared_delta += device_squared_delta(old2, new2);
            parity_delta += device_parity_delta(old1, new1);
            parity_delta += device_parity_delta(old2, new2);
        }
    } else {
        const int row1 = move.row1;
        const int row2 = move.row2;
        const int col1 = move.col1;
        const int col2 = move.col2;
        const int change1 = -2 * static_cast<int>(
            entries[static_cast<size_t>(col1) * params.order + row1]);
        const int change2 = -2 * static_cast<int>(
            entries[static_cast<size_t>(col2) * params.order + row2]);
        if (row1 == row2) {
            for (int other = lane; other < params.order;
                 other += k_warp_size) {
                if (other == row1) continue;
                const int old_value =
                    gram[static_cast<size_t>(row1) * params.order + other];
                const int new_value = old_value +
                    change1 * static_cast<int>(
                        entries[static_cast<size_t>(col1) *
                                params.order + other]) +
                    change2 * static_cast<int>(
                        entries[static_cast<size_t>(col2) *
                                params.order + other]);
                score_delta +=
                    device_residual_delta(old_value, new_value);
                conflict_delta +=
                    device_conflict_delta(old_value, new_value);
                squared_delta +=
                    device_squared_delta(old_value, new_value);
                parity_delta +=
                    device_parity_delta(old_value, new_value);
            }
        } else {
            for (int other = lane; other < params.order;
                 other += k_warp_size) {
                if (other == row1 || other == row2) continue;
                const int old1 =
                    gram[static_cast<size_t>(row1) * params.order + other];
                const int old2 =
                    gram[static_cast<size_t>(row2) * params.order + other];
                const int new1 = old1 +
                    change1 * static_cast<int>(
                        entries[static_cast<size_t>(col1) *
                                params.order + other]);
                const int new2 = old2 +
                    change2 * static_cast<int>(
                        entries[static_cast<size_t>(col2) *
                                params.order + other]);
                score_delta += device_residual_delta(old1, new1);
                score_delta += device_residual_delta(old2, new2);
                conflict_delta += device_conflict_delta(old1, new1);
                conflict_delta += device_conflict_delta(old2, new2);
                squared_delta += device_squared_delta(old1, new1);
                squared_delta += device_squared_delta(old2, new2);
                parity_delta += device_parity_delta(old1, new1);
                parity_delta += device_parity_delta(old2, new2);
            }
            if (lane == 0) {
                const int old_pair =
                    gram[static_cast<size_t>(row1) * params.order + row2];
                const int new_pair = old_pair +
                    change1 * static_cast<int>(
                        entries[static_cast<size_t>(col1) *
                                params.order + row2]) +
                    change2 * static_cast<int>(
                        entries[static_cast<size_t>(col2) *
                                params.order + row1]) +
                    (col1 == col2 ? change1 * change2 : 0);
                score_delta +=
                    device_residual_delta(old_pair, new_pair);
                conflict_delta +=
                    device_conflict_delta(old_pair, new_pair);
                squared_delta +=
                    device_squared_delta(old_pair, new_pair);
                parity_delta +=
                    device_parity_delta(old_pair, new_pair);
            }
        }
    }

    for (int offset = k_warp_size / 2; offset > 0; offset /= 2) {
        score_delta += __shfl_down_sync(0xffffffffu, score_delta, offset);
        conflict_delta += __shfl_down_sync(
            0xffffffffu, conflict_delta, offset);
        squared_delta += __shfl_down_sync(
            0xffffffffu, squared_delta, offset);
        parity_delta += __shfl_down_sync(
            0xffffffffu, parity_delta, offset);
    }
    return DeviceEvaluation{
        score_delta, conflict_delta, squared_delta, parity_delta
    };
}

__device__ long long reduce_warp(long long value) {
    for (int offset = k_warp_size / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ void commit_sign_flip(const DeviceMove& move,
                                 const DeviceEvaluation& evaluation,
                                 int8_t* entries,
                                 int16_t* gram,
                                 long long* row_residual,
                                 long long& score,
                                 long long& conflicts,
                                 const KernelParams& params) {
    const int row = move.row1;
    const int col = move.col1;
    const int change = -2 * static_cast<int>(
        entries[static_cast<size_t>(col) * params.order + row]);
    for (int other = threadIdx.x; other < params.order;
         other += blockDim.x) {
        if (other == row) continue;
        const size_t forward =
            static_cast<size_t>(row) * params.order + other;
        const size_t reverse =
            static_cast<size_t>(other) * params.order + row;
        const int old_value = gram[forward];
        const int new_value = old_value + change * static_cast<int>(
            entries[static_cast<size_t>(col) * params.order + other]);
        const long long delta = device_residual_delta(old_value, new_value);
        gram[forward] = static_cast<int16_t>(new_value);
        gram[reverse] = static_cast<int16_t>(new_value);
        row_residual[other] += delta;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        row_residual[row] += evaluation.score_delta;
        entries[static_cast<size_t>(col) * params.order + row] =
            static_cast<int8_t>(
                -entries[static_cast<size_t>(col) * params.order + row]);
        score += evaluation.score_delta;
        conflicts += evaluation.conflict_delta;
    }
    __syncthreads();
}

__device__ void commit_support_switch(const DeviceMove& move,
                                      const DeviceEvaluation& evaluation,
                                      int8_t* entries,
                                      int16_t* gram,
                                      long long* row_residual,
                                      uint16_t* minority,
                                      long long& score,
                                      long long& conflicts,
                                      const KernelParams& params,
                                      long long* row1_warp_delta,
                                      long long* row2_warp_delta) {
    const int row1 = move.row1;
    const int row2 = move.row2;
    const int col1 = move.col1;
    const int col2 = move.col2;
    const int old_value1 = static_cast<int>(
        entries[static_cast<size_t>(col1) * params.order + row1]);
    const int old_value2 = static_cast<int>(
        entries[static_cast<size_t>(col2) * params.order + row2]);
    long long row1_delta = 0;
    long long row2_delta = 0;

    for (int other = threadIdx.x; other < params.order;
         other += blockDim.x) {
        if (other == row1 || other == row2) continue;
        const size_t row1_forward =
            static_cast<size_t>(row1) * params.order + other;
        const size_t row1_reverse =
            static_cast<size_t>(other) * params.order + row1;
        const size_t row2_forward =
            static_cast<size_t>(row2) * params.order + other;
        const size_t row2_reverse =
            static_cast<size_t>(other) * params.order + row2;
        const int old1 = gram[row1_forward];
        const int old2 = gram[row2_forward];
        const int new1 = old1 +
            move.value1 * static_cast<int>(
                entries[static_cast<size_t>(col2) * params.order + other]) -
            old_value1 * static_cast<int>(
                entries[static_cast<size_t>(col1) * params.order + other]);
        const int new2 = old2 +
            move.value2 * static_cast<int>(
                entries[static_cast<size_t>(col1) * params.order + other]) -
            old_value2 * static_cast<int>(
                entries[static_cast<size_t>(col2) * params.order + other]);
        const long long delta1 = device_residual_delta(old1, new1);
        const long long delta2 = device_residual_delta(old2, new2);
        gram[row1_forward] = static_cast<int16_t>(new1);
        gram[row1_reverse] = static_cast<int16_t>(new1);
        gram[row2_forward] = static_cast<int16_t>(new2);
        gram[row2_reverse] = static_cast<int16_t>(new2);
        row_residual[other] += delta1 + delta2;
        row1_delta += delta1;
        row2_delta += delta2;
    }

    const int lane = threadIdx.x % k_warp_size;
    const int warp = threadIdx.x / k_warp_size;
    row1_delta = reduce_warp(row1_delta);
    row2_delta = reduce_warp(row2_delta);
    if (lane == 0) {
        row1_warp_delta[warp] = row1_delta;
        row2_warp_delta[warp] = row2_delta;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        long long total1 = 0;
        long long total2 = 0;
        for (int i = 0; i < k_warps_per_block; ++i) {
            total1 += row1_warp_delta[i];
            total2 += row2_warp_delta[i];
        }
        row_residual[row1] += total1;
        row_residual[row2] += total2;
        entries[static_cast<size_t>(col1) * params.order + row1] = 0;
        entries[static_cast<size_t>(col2) * params.order + row2] = 0;
        entries[static_cast<size_t>(col2) * params.order + row1] =
            static_cast<int8_t>(move.value1);
        entries[static_cast<size_t>(col1) * params.order + row2] =
            static_cast<int8_t>(move.value2);
        if (params.sparse_side != 0) {
            minority[static_cast<size_t>(row1) * params.minority_count +
                     move.minority_pos1] = static_cast<uint16_t>(col2);
            minority[static_cast<size_t>(row2) * params.minority_count +
                     move.minority_pos2] = static_cast<uint16_t>(col1);
        } else {
            minority[static_cast<size_t>(row1) * params.minority_count +
                     move.minority_pos1] = static_cast<uint16_t>(col1);
            minority[static_cast<size_t>(row2) * params.minority_count +
                     move.minority_pos2] = static_cast<uint16_t>(col2);
        }
        score += evaluation.score_delta;
        conflicts += evaluation.conflict_delta;
    }
    __syncthreads();
}

__device__ void commit_double_sign_flip(
    const DeviceMove& move,
    const DeviceEvaluation& evaluation,
    int8_t* entries,
    int16_t* gram,
    long long* row_residual,
    long long& score,
    long long& conflicts,
    const KernelParams& params,
    long long* row1_warp_delta,
    long long* row2_warp_delta) {
    const int row1 = move.row1;
    const int row2 = move.row2;
    const int col1 = move.col1;
    const int col2 = move.col2;
    const int change1 = -2 * static_cast<int>(
        entries[static_cast<size_t>(col1) * params.order + row1]);
    const int change2 = -2 * static_cast<int>(
        entries[static_cast<size_t>(col2) * params.order + row2]);
    long long row1_delta = 0;
    long long row2_delta = 0;

    if (row1 == row2) {
        for (int other = threadIdx.x; other < params.order;
             other += blockDim.x) {
            if (other == row1) continue;
            const size_t forward =
                static_cast<size_t>(row1) * params.order + other;
            const size_t reverse =
                static_cast<size_t>(other) * params.order + row1;
            const int old_value = gram[forward];
            const int new_value = old_value +
                change1 * static_cast<int>(
                    entries[static_cast<size_t>(col1) *
                            params.order + other]) +
                change2 * static_cast<int>(
                    entries[static_cast<size_t>(col2) *
                            params.order + other]);
            const long long delta =
                device_residual_delta(old_value, new_value);
            gram[forward] = static_cast<int16_t>(new_value);
            gram[reverse] = static_cast<int16_t>(new_value);
            row_residual[other] += delta;
            row1_delta += delta;
        }
    } else {
        for (int other = threadIdx.x; other < params.order;
             other += blockDim.x) {
            if (other == row1 || other == row2) continue;
            const size_t row1_forward =
                static_cast<size_t>(row1) * params.order + other;
            const size_t row1_reverse =
                static_cast<size_t>(other) * params.order + row1;
            const size_t row2_forward =
                static_cast<size_t>(row2) * params.order + other;
            const size_t row2_reverse =
                static_cast<size_t>(other) * params.order + row2;
            const int old1 = gram[row1_forward];
            const int old2 = gram[row2_forward];
            const int new1 = old1 +
                change1 * static_cast<int>(
                    entries[static_cast<size_t>(col1) *
                            params.order + other]);
            const int new2 = old2 +
                change2 * static_cast<int>(
                    entries[static_cast<size_t>(col2) *
                            params.order + other]);
            const long long delta1 =
                device_residual_delta(old1, new1);
            const long long delta2 =
                device_residual_delta(old2, new2);
            gram[row1_forward] = static_cast<int16_t>(new1);
            gram[row1_reverse] = static_cast<int16_t>(new1);
            gram[row2_forward] = static_cast<int16_t>(new2);
            gram[row2_reverse] = static_cast<int16_t>(new2);
            row_residual[other] += delta1 + delta2;
            row1_delta += delta1;
            row2_delta += delta2;
        }
    }

    const int lane = threadIdx.x % k_warp_size;
    const int warp = threadIdx.x / k_warp_size;
    row1_delta = reduce_warp(row1_delta);
    row2_delta = reduce_warp(row2_delta);
    if (lane == 0) {
        row1_warp_delta[warp] = row1_delta;
        row2_warp_delta[warp] = row2_delta;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        long long total1 = 0;
        long long total2 = 0;
        for (int index = 0; index < k_warps_per_block; ++index) {
            total1 += row1_warp_delta[index];
            total2 += row2_warp_delta[index];
        }
        if (row1 == row2) {
            row_residual[row1] += total1;
        } else {
            const size_t forward =
                static_cast<size_t>(row1) * params.order + row2;
            const size_t reverse =
                static_cast<size_t>(row2) * params.order + row1;
            const int old_pair = gram[forward];
            const int new_pair = old_pair +
                change1 * static_cast<int>(
                    entries[static_cast<size_t>(col1) *
                            params.order + row2]) +
                change2 * static_cast<int>(
                    entries[static_cast<size_t>(col2) *
                            params.order + row1]) +
                (col1 == col2 ? change1 * change2 : 0);
            const long long pair_delta =
                device_residual_delta(old_pair, new_pair);
            gram[forward] = static_cast<int16_t>(new_pair);
            gram[reverse] = static_cast<int16_t>(new_pair);
            row_residual[row1] += total1 + pair_delta;
            row_residual[row2] += total2 + pair_delta;
        }
        entries[static_cast<size_t>(col1) * params.order + row1] =
            static_cast<int8_t>(
                -entries[static_cast<size_t>(col1) *
                         params.order + row1]);
        entries[static_cast<size_t>(col2) * params.order + row2] =
            static_cast<int8_t>(
                -entries[static_cast<size_t>(col2) *
                         params.order + row2]);
        score += evaluation.score_delta;
        conflicts += evaluation.conflict_delta;
    }
    __syncthreads();
}

__global__ void initialize_rng_kernel(curandStatePhilox4_32_10_t* states,
                                      int replicas,
                                      unsigned long long seed) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= replicas) return;
    curand_init(seed, static_cast<unsigned long long>(index), 0, &states[index]);
}

__global__ void anneal_kernel(DeviceStorage storage, KernelParams params) {
    extern __shared__ __align__(16) unsigned char dynamic_shared[];

    const size_t matrix_items =
        static_cast<size_t>(params.order) * params.order;
    const size_t minority_items =
        static_cast<size_t>(params.order) * params.minority_count;
    size_t offset = matrix_items * sizeof(int8_t);
    const size_t gram_offset = device_align_up(offset, alignof(int16_t));
    offset = gram_offset + matrix_items * sizeof(int16_t);
    const size_t residual_offset = device_align_up(offset, alignof(long long));
    offset = residual_offset +
             static_cast<size_t>(params.order) * sizeof(long long);
    const size_t minority_offset = device_align_up(offset, alignof(uint16_t));

    int8_t* entries = reinterpret_cast<int8_t*>(dynamic_shared);
    int16_t* gram =
        reinterpret_cast<int16_t*>(dynamic_shared + gram_offset);
    long long* row_residual =
        reinterpret_cast<long long*>(dynamic_shared + residual_offset);
    uint16_t* minority =
        reinterpret_cast<uint16_t*>(dynamic_shared + minority_offset);

    __shared__ DeviceMove wave_moves[k_warps_per_block];
    __shared__ DeviceEvaluation wave_evaluations[k_warps_per_block];
    __shared__ int wave_valid[k_warps_per_block];
    __shared__ DeviceMove selected_move;
    __shared__ DeviceEvaluation selected_evaluation;
    __shared__ long long row1_warp_delta[k_warps_per_block];
    __shared__ long long row2_warp_delta[k_warps_per_block];
    __shared__ curandStatePhilox4_32_10_t rng;
    __shared__ long long current_score;
    __shared__ long long current_conflicts;
    __shared__ long long best_score;
    __shared__ long long best_conflicts;
    __shared__ unsigned long long move_count;
    __shared__ unsigned long long candidate_count;
    __shared__ double temperature;
    __shared__ double reheat_temperature;
    __shared__ int moves_since_improvement;
    __shared__ int cooling_counter;
    __shared__ int sample_count;
    __shared__ int wave_count;
    __shared__ int have_move;
    __shared__ int accept_move;
    __shared__ int save_best;
    __shared__ int stop_replica;
    __shared__ int objective_kind;

    const size_t replica = blockIdx.x;
    const size_t matrix_base = replica * matrix_items;
    const size_t residual_base =
        replica * static_cast<size_t>(params.order);
    const size_t minority_base = replica * minority_items;

    for (size_t index = threadIdx.x; index < matrix_items;
         index += blockDim.x) {
        entries[index] = storage.current_entries[matrix_base + index];
        gram[index] = storage.current_gram[matrix_base + index];
    }
    for (int row = threadIdx.x; row < params.order; row += blockDim.x)
        row_residual[row] = storage.current_residual[residual_base + row];
    if (minority_items != 0) {
        for (size_t index = threadIdx.x; index < minority_items;
             index += blockDim.x) {
            minority[index] =
                storage.current_minority[minority_base + index];
        }
    }
    if (threadIdx.x == 0) {
        current_score = storage.current_score[replica];
        current_conflicts = storage.current_conflicts[replica];
        best_score = storage.best_score[replica];
        best_conflicts = storage.best_conflicts[replica];
        temperature = storage.temperature[replica];
        reheat_temperature = storage.reheat_temperature[replica];
        moves_since_improvement =
            storage.moves_since_improvement[replica];
        cooling_counter = storage.cooling_counter[replica];
        rng = storage.rng[replica];
        move_count = storage.moves[replica];
        candidate_count = storage.candidate_evaluations[replica];
        const uint32_t objective_hash =
            static_cast<uint32_t>(replica) * UINT32_C(0x9e3779b9) +
            UINT32_C(0x85ebca6b);
        const double objective_choice =
            static_cast<double>(objective_hash) *
            (1.0 / 4294967296.0);
        objective_kind =
            objective_choice < params.squared_objective_fraction
                ? 1
                : objective_choice <
                    params.squared_objective_fraction +
                        params.parity_objective_fraction
                    ? 2 : 0;
    }
    __syncthreads();

    const int lane = threadIdx.x % k_warp_size;
    const int warp = threadIdx.x / k_warp_size;

    for (int iteration = 0; iteration < params.moves_per_launch; ++iteration) {
        if (threadIdx.x == 0) {
            stop_replica = best_score == 0 ||
                ((iteration & 31) == 0 && atomicAdd(storage.solved, 0) != 0);
        }
        __syncthreads();
        if (stop_replica != 0) break;

        if (threadIdx.x == 0) {
            sample_count =
                random_unit(rng) < params.greedy_fraction
                    ? params.candidate_samples
                    : 1;
            have_move = 0;
        }
        __syncthreads();

        for (int wave_start = 0;
             wave_start < xweigh_cuda::MAX_CANDIDATE_SAMPLES;
             wave_start += k_warps_per_block) {
            if (threadIdx.x == 0) {
                wave_count = sample_count - wave_start;
                if (wave_count > k_warps_per_block)
                    wave_count = k_warps_per_block;
                if (wave_count < 0) wave_count = 0;
                for (int candidate = 0; candidate < wave_count; ++candidate) {
                    wave_valid[candidate] = propose_move(
                        rng, entries, row_residual, minority, params,
                        wave_moves[candidate]) ? 1 : 0;
                }
            }
            __syncthreads();
            if (wave_count == 0) break;

            if (warp < wave_count && wave_valid[warp] != 0) {
                const DeviceEvaluation evaluation = evaluate_move_warp(
                    wave_moves[warp], entries, gram, params, lane);
                if (lane == 0) wave_evaluations[warp] = evaluation;
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                for (int candidate = 0; candidate < wave_count; ++candidate) {
                    if (wave_valid[candidate] == 0) continue;
                    ++candidate_count;
                    if (have_move == 0 ||
                        device_better_evaluation(
                            wave_evaluations[candidate],
                            selected_evaluation,
                            objective_kind)) {
                        selected_move = wave_moves[candidate];
                        selected_evaluation = wave_evaluations[candidate];
                        have_move = 1;
                    }
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            accept_move = 0;
            if (have_move != 0) {
                const long long objective_delta =
                    objective_kind == 1
                        ? selected_evaluation.squared_delta
                    : objective_kind == 2
                        ? selected_evaluation.parity_delta
                        : selected_evaluation.score_delta;
                accept_move =
                    objective_delta <= 0 ||
                    random_unit(rng) <
                        exp(-static_cast<double>(
                                objective_delta) /
                            (temperature > 1.0e-12
                                 ? temperature
                                 : 1.0e-12));
            }
        }
        __syncthreads();

        if (have_move != 0 && accept_move != 0) {
            if (selected_move.kind == k_sign_flip) {
                commit_sign_flip(
                    selected_move, selected_evaluation, entries, gram,
                    row_residual, current_score, current_conflicts, params);
            } else if (selected_move.kind == k_support_switch) {
                commit_support_switch(
                    selected_move, selected_evaluation, entries, gram,
                    row_residual, minority, current_score, current_conflicts,
                    params, row1_warp_delta, row2_warp_delta);
            } else {
                commit_double_sign_flip(
                    selected_move, selected_evaluation, entries, gram,
                    row_residual, current_score, current_conflicts, params,
                    row1_warp_delta, row2_warp_delta);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            save_best = 0;
            if (have_move != 0) {
                ++move_count;
                if (device_better_pair(current_score, current_conflicts,
                                       best_score, best_conflicts)) {
                    best_score = current_score;
                    best_conflicts = current_conflicts;
                    moves_since_improvement = 0;
                    save_best = 1;
                } else {
                    ++moves_since_improvement;
                }

                if (++cooling_counter >= params.moves_per_cool) {
                    cooling_counter = 0;
                    temperature *= params.cooling;
                    if (temperature < params.minimum_temperature)
                        temperature = params.minimum_temperature;
                }
                if (moves_since_improvement >= params.stuck_threshold) {
                    moves_since_improvement = 0;
                    temperature = reheat_temperature * params.reheat;
                    if (temperature < params.minimum_temperature)
                        temperature = params.minimum_temperature;
                }
            }
        }
        __syncthreads();

        if (save_best != 0) {
            for (size_t index = threadIdx.x; index < matrix_items;
                 index += blockDim.x) {
                storage.best_entries[matrix_base + index] = entries[index];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && save_best != 0) {
            storage.best_score[replica] = best_score;
            storage.best_conflicts[replica] = best_conflicts;
            if (best_score == 0) atomicExch(storage.solved, 1);
        }
        __syncthreads();
    }

    for (size_t index = threadIdx.x; index < matrix_items;
         index += blockDim.x) {
        storage.current_entries[matrix_base + index] = entries[index];
        storage.current_gram[matrix_base + index] = gram[index];
    }
    for (int row = threadIdx.x; row < params.order; row += blockDim.x)
        storage.current_residual[residual_base + row] = row_residual[row];
    if (minority_items != 0) {
        for (size_t index = threadIdx.x; index < minority_items;
             index += blockDim.x) {
            storage.current_minority[minority_base + index] =
                minority[index];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        storage.current_score[replica] = current_score;
        storage.current_conflicts[replica] = current_conflicts;
        storage.temperature[replica] = temperature;
        storage.moves_since_improvement[replica] =
            moves_since_improvement;
        storage.cooling_counter[replica] = cooling_counter;
        storage.rng[replica] = rng;
        storage.moves[replica] = move_count;
        storage.candidate_evaluations[replica] = candidate_count;
    }
}

void validate_params(int order, int weight, const xweigh_cuda::Params& params) {
    if (order < 1 || order > xweigh::MAX_ORDER)
        throw std::runtime_error("CUDA xweigh order is out of range");
    if (weight < 1 || weight > order)
        throw std::runtime_error("CUDA xweigh weight must be in [1, n]");
    if (params.replicas < 0)
        throw std::runtime_error("CUDA replica count must be nonnegative");
    if (params.moves_per_launch < 1)
        throw std::runtime_error("CUDA moves per launch must be positive");
    if (params.max_seconds < 0.0)
        throw std::runtime_error("CUDA max seconds must be nonnegative");
    if (params.save_interval < 0.0)
        throw std::runtime_error("CUDA save interval must be nonnegative");
    if (params.sign_fraction < 0.0 || params.sign_fraction > 1.0)
        throw std::runtime_error("CUDA sign fraction must be in [0, 1]");
    if (params.double_sign_fraction < 0.0 ||
        params.double_sign_fraction > 1.0) {
        throw std::runtime_error(
            "CUDA double-sign fraction must be in [0, 1]");
    }
    if (params.switch_sign_fraction < 0.0 ||
        params.switch_sign_fraction > 1.0) {
        throw std::runtime_error(
            "CUDA switch-sign fraction must be in [0, 1]");
    }
    if (params.squared_objective_fraction < 0.0 ||
        params.squared_objective_fraction > 1.0) {
        throw std::runtime_error(
            "CUDA squared-objective fraction must be in [0, 1]");
    }
    if (params.parity_objective_fraction < 0.0 ||
        params.parity_objective_fraction > 1.0) {
        throw std::runtime_error(
            "CUDA parity-objective fraction must be in [0, 1]");
    }
    if (params.squared_objective_fraction +
            params.parity_objective_fraction >
        1.0) {
        throw std::runtime_error(
            "CUDA alternate objective fractions must sum to at most 1");
    }
    if (params.greedy_fraction < 0.0 || params.greedy_fraction > 1.0)
        throw std::runtime_error("CUDA greedy fraction must be in [0, 1]");
    if (params.target_fraction < 0.0 || params.target_fraction > 1.0)
        throw std::runtime_error("CUDA target fraction must be in [0, 1]");
    if (params.candidate_samples < 1 ||
        params.candidate_samples > xweigh_cuda::MAX_CANDIDATE_SAMPLES) {
        throw std::runtime_error(
            "CUDA candidate samples must be in [1, MAX_CANDIDATE_SAMPLES]");
    }
    if (params.target_samples < 1 ||
        params.target_samples > xweigh_cuda::MAX_TARGET_SAMPLES) {
        throw std::runtime_error(
            "CUDA target samples must be in [1, MAX_TARGET_SAMPLES]");
    }
    if (params.t_init < 0.0)
        throw std::runtime_error("CUDA initial temperature must be nonnegative");
    if (params.t_min <= 0.0)
        throw std::runtime_error("CUDA minimum temperature must be positive");
    if (params.cooling <= 0.0 || params.cooling >= 1.0)
        throw std::runtime_error("CUDA cooling factor must be in (0, 1)");
    if (params.moves_per_cool < 1)
        throw std::runtime_error("CUDA moves per cool must be positive");
    if (params.stuck_threshold < 1)
        throw std::runtime_error("CUDA stuck threshold must be positive");
    if (params.reheat <= 0.0)
        throw std::runtime_error("CUDA reheat factor must be positive");
    if (params.restart_interval < 0.0)
        throw std::runtime_error(
            "CUDA restart interval must be nonnegative");
    if (params.restart_fraction <= 0.0 ||
        params.restart_fraction > 1.0) {
        throw std::runtime_error(
            "CUDA restart fraction must be in (0, 1]");
    }
    if (params.restart_kick_min < 0)
        throw std::runtime_error(
            "CUDA minimum restart kick must be nonnegative");
    if (params.restart_kick_max < params.restart_kick_min)
        throw std::runtime_error(
            "CUDA maximum restart kick must be at least the minimum");
}

xweigh::Params calibration_params(const xweigh_cuda::Params& params) {
    xweigh::Params result;
    result.sign_fraction = params.sign_fraction;
    result.greedy_fraction = params.greedy_fraction;
    result.candidate_samples = params.candidate_samples;
    result.target_fraction = params.target_fraction;
    result.target_samples = params.target_samples;
    result.t_init = params.t_init;
    result.t_min = params.t_min;
    result.cooling = params.cooling;
    result.moves_per_cool = params.moves_per_cool;
    result.stuck_threshold = params.stuck_threshold;
    result.reheat = params.reheat;
    return result;
}

struct PackedScore {
    long long score = 0;
    long long conflicts = 0;
};

PackedScore pack_state(const xweigh::State& state,
                       int replica,
                       int order,
                       int weight,
                       int minority_count,
                       bool sparse_side,
                       std::vector<int8_t>& entries,
                       std::vector<int16_t>& gram,
                       std::vector<long long>& row_residual,
                       std::vector<uint16_t>& minority) {
    const size_t n = static_cast<size_t>(order);
    const size_t matrix_items = n * n;
    const size_t matrix_base = static_cast<size_t>(replica) * matrix_items;
    const size_t residual_base = static_cast<size_t>(replica) * n;
    const size_t minority_base =
        static_cast<size_t>(replica) * n * minority_count;

    std::copy(state.entries().begin(), state.entries().end(),
              entries.begin() + static_cast<std::ptrdiff_t>(matrix_base));

    if (minority_count != 0) {
        for (int row = 0; row < order; ++row) {
            int position = 0;
            for (int col = 0; col < order; ++col) {
                const bool nonzero =
                    entries[matrix_base +
                            static_cast<size_t>(col) * order + row] != 0;
                if (nonzero == sparse_side) {
                    if (position >= minority_count)
                        throw std::runtime_error(
                            "CUDA initial support is not regular");
                    minority[minority_base +
                             static_cast<size_t>(row) * minority_count +
                             position] = static_cast<uint16_t>(col);
                    ++position;
                }
            }
            if (position != minority_count)
                throw std::runtime_error(
                    "CUDA initial support is not regular");
        }
    }

    PackedScore packed;
    for (int row1 = 0; row1 < order; ++row1) {
        gram[matrix_base + static_cast<size_t>(row1) * order + row1] =
            static_cast<int16_t>(weight);
        for (int row2 = row1 + 1; row2 < order; ++row2) {
            int dot = 0;
            for (int col = 0; col < order; ++col) {
                dot += static_cast<int>(
                    entries[matrix_base +
                            static_cast<size_t>(col) * order + row1]) *
                       static_cast<int>(
                    entries[matrix_base +
                            static_cast<size_t>(col) * order + row2]);
            }
            gram[matrix_base + static_cast<size_t>(row1) * order + row2] =
                static_cast<int16_t>(dot);
            gram[matrix_base + static_cast<size_t>(row2) * order + row1] =
                static_cast<int16_t>(dot);
            const long long residual =
                static_cast<long long>(dot < 0 ? -dot : dot);
            row_residual[residual_base + row1] += residual;
            row_residual[residual_base + row2] += residual;
            packed.score += residual;
            if (dot != 0) ++packed.conflicts;
        }
    }

    if (packed.score != state.score() ||
        packed.conflicts != state.conflicts()) {
        throw std::runtime_error(
            "CUDA initial packed-state score mismatch");
    }
    return packed;
}

int select_best_replica(const std::vector<long long>& scores,
                        const std::vector<long long>& conflicts) {
    int best = 0;
    for (int replica = 1; replica < static_cast<int>(scores.size());
         ++replica) {
        if (xweigh::better_pair(
                scores[replica], conflicts[replica],
                scores[best], conflicts[best])) {
            best = replica;
        }
    }
    return best;
}

uint64_t sum_counters(const std::vector<unsigned long long>& counters) {
    uint64_t total = 0;
    for (const unsigned long long value : counters)
        total += static_cast<uint64_t>(value);
    return total;
}

std::vector<int8_t> copy_best_entries(const DeviceBuffer<int8_t>& best_entries,
                                      int replica,
                                      size_t matrix_items) {
    std::vector<int8_t> result(matrix_items);
    check_cuda(
        cudaMemcpy(
            result.data(),
            best_entries.data() +
                static_cast<size_t>(replica) * matrix_items,
            checked_multiply(matrix_items, sizeof(int8_t), "best entries"),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy best entries to host");
    return result;
}

}  // namespace

namespace xweigh_cuda {

DeviceInfo query_device(int ordinal) {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (ordinal < 0 || ordinal >= device_count) {
        std::ostringstream message;
        message << "CUDA device ordinal " << ordinal
                << " is out of range; found " << device_count << " device(s)";
        throw std::runtime_error(message.str());
    }

    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, ordinal),
        "cudaGetDeviceProperties");
    int opt_in_shared_memory = 0;
    check_cuda(
        cudaDeviceGetAttribute(
            &opt_in_shared_memory,
            cudaDevAttrMaxSharedMemoryPerBlockOptin,
            ordinal),
        "cudaDeviceGetAttribute max opt-in shared memory");
    if (opt_in_shared_memory <= 0)
        opt_in_shared_memory =
            static_cast<int>(properties.sharedMemPerBlock);

    DeviceInfo result;
    result.name = properties.name;
    result.ordinal = ordinal;
    result.compute_major = properties.major;
    result.compute_minor = properties.minor;
    result.multiprocessors = properties.multiProcessorCount;
    result.global_memory = properties.totalGlobalMem;
    result.max_shared_memory_per_block =
        static_cast<size_t>(opt_in_shared_memory);
    return result;
}

RunResult run_annealer(
    int order, int weight, const Params& params,
    const xweigh::State* initial_state) {
    validate_params(order, weight, params);
    if (initial_state != nullptr) {
        if (initial_state->order() != order ||
            initial_state->weight() != weight) {
            throw std::runtime_error(
                "CUDA initial state dimensions do not match W(n,w)");
        }
        if (!initial_state->verify_support())
            throw std::runtime_error(
                "CUDA initial state is not fixed-weight");
    }
    const DeviceInfo device = query_device(params.device);
    check_cuda(cudaSetDevice(params.device), "cudaSetDevice");

    const int minority_count = std::min(weight, order - weight);
    const bool sparse_side = weight <= order - weight;
    const SharedLayout shared_layout =
        make_shared_layout(order, minority_count);

    cudaFuncAttributes kernel_attributes{};
    check_cuda(
        cudaFuncGetAttributes(&kernel_attributes, anneal_kernel),
        "cudaFuncGetAttributes anneal kernel");
    const size_t static_shared_bytes =
        static_cast<size_t>(kernel_attributes.sharedSizeBytes);
    const size_t total_shared_bytes = checked_add(
        shared_layout.bytes, static_shared_bytes, "total shared memory");
    if (total_shared_bytes > device.max_shared_memory_per_block) {
        std::ostringstream message;
        message << "CUDA replica requires "
                << std::fixed << std::setprecision(2)
                << static_cast<double>(total_shared_bytes) / 1024.0
                << " KiB of shared memory, but device " << device.ordinal
                << " permits "
                << static_cast<double>(device.max_shared_memory_per_block) /
                       1024.0
                << " KiB per opt-in block";
        throw std::runtime_error(message.str());
    }
    if (shared_layout.bytes >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "CUDA dynamic shared-memory request exceeds the runtime limit");
    }
    check_cuda(
        cudaFuncSetAttribute(
            anneal_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_layout.bytes)),
        "cudaFuncSetAttribute max dynamic shared memory");

    int resident_blocks_per_multiprocessor = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks_per_multiprocessor,
            anneal_kernel,
            k_block_threads,
            shared_layout.bytes),
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    if (resident_blocks_per_multiprocessor < 1)
        throw std::runtime_error(
            "CUDA anneal kernel has zero occupancy on the selected device");

    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, params.device),
        "cudaGetDeviceProperties for launch limits");
    const long long default_replicas =
        static_cast<long long>(resident_blocks_per_multiprocessor) *
        device.multiprocessors * 2;
    const long long requested_replicas =
        params.replicas == 0 ? default_replicas : params.replicas;
    if (requested_replicas < 1 ||
        requested_replicas > properties.maxGridSize[0] ||
        requested_replicas > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "CUDA replica count exceeds the device grid limit");
    }
    const int replicas = static_cast<int>(requested_replicas);

    std::cout << "[cuda] device=" << device.name
              << " compute=" << device.compute_major << '.'
              << device.compute_minor
              << " multiprocessors=" << device.multiprocessors
              << " replicas=" << replicas
              << " shared_kib="
              << static_cast<double>(total_shared_bytes) / 1024.0
              << '\n';

    const auto initialization_start = std::chrono::steady_clock::now();
    const size_t n = static_cast<size_t>(order);
    const size_t matrix_items = checked_multiply(n, n, "matrix");
    const size_t minority_items = checked_multiply(
        n, static_cast<size_t>(minority_count), "minority");
    const size_t total_matrix_items = checked_multiply(
        matrix_items, static_cast<size_t>(replicas), "replica matrices");
    const size_t total_residual_items = checked_multiply(
        n, static_cast<size_t>(replicas), "replica residuals");
    const size_t total_minority_items = checked_multiply(
        minority_items, static_cast<size_t>(replicas), "replica minority");

    std::vector<int8_t> host_entries(total_matrix_items);
    std::vector<int8_t> host_best_entries(total_matrix_items);
    std::vector<int16_t> host_gram(total_matrix_items);
    std::vector<long long> host_residual(total_residual_items);
    std::vector<uint16_t> host_minority(total_minority_items);
    std::vector<long long> host_current_scores(
        static_cast<size_t>(replicas));
    std::vector<long long> host_current_conflicts(
        static_cast<size_t>(replicas));
    std::vector<long long> host_best_scores(
        static_cast<size_t>(replicas));
    std::vector<long long> host_best_conflicts(
        static_cast<size_t>(replicas));
    std::vector<double> host_temperatures(
        static_cast<size_t>(replicas));
    std::vector<double> host_reheat_temperatures(
        static_cast<size_t>(replicas));
    std::vector<int> host_moves_since_improvement(
        static_cast<size_t>(replicas), 0);
    std::vector<int> host_cooling_counters(
        static_cast<size_t>(replicas), 0);
    std::vector<unsigned long long> host_moves(
        static_cast<size_t>(replicas), 0);
    std::vector<unsigned long long> host_candidate_evaluations(
        static_cast<size_t>(replicas), 0);

    const uint32_t base_seed = params.seed != 0
        ? static_cast<uint32_t>(params.seed)
        : static_cast<uint32_t>(std::random_device{}());
    const xweigh::Params cpu_params = calibration_params(params);
    double initial_temperature = 0.0;

    for (int replica = 0; replica < replicas; ++replica) {
        const uint32_t state_seed =
            base_seed +
            static_cast<uint32_t>(replica) * UINT32_C(0x9e3779b9);
        xweigh::State state = initial_state != nullptr
            ? *initial_state
            : xweigh::State::random_start(
                order, weight, state_seed);
        if (replica == 0) {
            initial_temperature =
                xweigh::calibrate_temperature(
                    state, cpu_params, state_seed);
        }
        const PackedScore packed = pack_state(
            state, replica, order, weight, minority_count, sparse_side,
            host_entries, host_gram, host_residual, host_minority);
        host_current_scores[static_cast<size_t>(replica)] = packed.score;
        host_current_conflicts[static_cast<size_t>(replica)] =
            packed.conflicts;
        host_best_scores[static_cast<size_t>(replica)] = packed.score;
        host_best_conflicts[static_cast<size_t>(replica)] =
            packed.conflicts;
    }
    host_best_entries = host_entries;

    const double bottom_temperature =
        std::min(params.t_min, initial_temperature);
    const double temperature_ratio = replicas > 1
        ? std::pow(
            initial_temperature / std::max(bottom_temperature, 1.0e-12),
            1.0 / static_cast<double>(replicas - 1))
        : 1.0;
    double replica_temperature = bottom_temperature;
    for (int replica = 0; replica < replicas; ++replica) {
        host_temperatures[static_cast<size_t>(replica)] =
            replica_temperature;
        host_reheat_temperatures[static_cast<size_t>(replica)] =
            replica_temperature;
        replica_temperature *= temperature_ratio;
    }

    size_t replica_bytes = 0;
    replica_bytes = checked_add(
        replica_bytes,
        checked_multiply(matrix_items, sizeof(int8_t), "current entries"),
        "replica");
    replica_bytes = checked_add(
        replica_bytes,
        checked_multiply(matrix_items, sizeof(int16_t), "current Gram"),
        "replica");
    replica_bytes = checked_add(
        replica_bytes,
        checked_multiply(n, sizeof(long long), "current residual"),
        "replica");
    replica_bytes = checked_add(
        replica_bytes,
        checked_multiply(minority_items, sizeof(uint16_t), "current minority"),
        "replica");
    replica_bytes = checked_add(
        replica_bytes,
        checked_multiply(matrix_items, sizeof(int8_t), "best entries"),
        "replica");
    replica_bytes = checked_add(
        replica_bytes,
        4 * sizeof(long long) + 2 * sizeof(double) + 2 * sizeof(int) +
            sizeof(curandStatePhilox4_32_10_t) +
            2 * sizeof(unsigned long long),
        "replica");
    const size_t total_device_bytes = checked_add(
        checked_multiply(
            replica_bytes, static_cast<size_t>(replicas), "all replicas"),
        sizeof(int),
        "device state");

    size_t free_device_bytes = 0;
    size_t total_device_memory = 0;
    check_cuda(
        cudaMemGetInfo(&free_device_bytes, &total_device_memory),
        "cudaMemGetInfo");
    if (total_device_bytes > free_device_bytes) {
        std::ostringstream message;
        message << "CUDA replica state requires "
                << static_cast<double>(total_device_bytes) /
                       (1024.0 * 1024.0)
                << " MiB, but only "
                << static_cast<double>(free_device_bytes) /
                       (1024.0 * 1024.0)
                << " MiB is free";
        throw std::runtime_error(message.str());
    }

    DeviceBuffer<int8_t> current_entries(total_matrix_items);
    DeviceBuffer<int16_t> current_gram(total_matrix_items);
    DeviceBuffer<long long> current_residual(total_residual_items);
    DeviceBuffer<uint16_t> current_minority(total_minority_items);
    DeviceBuffer<long long> current_scores(static_cast<size_t>(replicas));
    DeviceBuffer<long long> current_conflicts(static_cast<size_t>(replicas));
    DeviceBuffer<int8_t> best_entries(total_matrix_items);
    DeviceBuffer<long long> best_scores(static_cast<size_t>(replicas));
    DeviceBuffer<long long> best_conflicts(static_cast<size_t>(replicas));
    DeviceBuffer<double> temperatures(static_cast<size_t>(replicas));
    DeviceBuffer<double> reheat_temperatures(static_cast<size_t>(replicas));
    DeviceBuffer<int> moves_since_improvement(static_cast<size_t>(replicas));
    DeviceBuffer<int> cooling_counters(static_cast<size_t>(replicas));
    DeviceBuffer<curandStatePhilox4_32_10_t> rng_states(
        static_cast<size_t>(replicas));
    DeviceBuffer<unsigned long long> move_counters(
        static_cast<size_t>(replicas));
    DeviceBuffer<unsigned long long> candidate_counters(
        static_cast<size_t>(replicas));
    DeviceBuffer<int> solved(1);

    copy_to_device(
        current_entries, host_entries, "cudaMemcpy current entries to device");
    copy_to_device(
        current_gram, host_gram, "cudaMemcpy current Gram to device");
    copy_to_device(
        current_residual, host_residual,
        "cudaMemcpy current residual to device");
    copy_to_device(
        current_minority, host_minority,
        "cudaMemcpy current minority to device");
    copy_to_device(
        current_scores, host_current_scores,
        "cudaMemcpy current scores to device");
    copy_to_device(
        current_conflicts, host_current_conflicts,
        "cudaMemcpy current conflicts to device");
    copy_to_device(
        best_entries, host_best_entries,
        "cudaMemcpy best entries to device");
    copy_to_device(
        best_scores, host_best_scores, "cudaMemcpy best scores to device");
    copy_to_device(
        best_conflicts, host_best_conflicts,
        "cudaMemcpy best conflicts to device");
    copy_to_device(
        temperatures, host_temperatures,
        "cudaMemcpy temperatures to device");
    copy_to_device(
        reheat_temperatures, host_reheat_temperatures,
        "cudaMemcpy reheat temperatures to device");
    copy_to_device(
        moves_since_improvement, host_moves_since_improvement,
        "cudaMemcpy improvement counters to device");
    copy_to_device(
        cooling_counters, host_cooling_counters,
        "cudaMemcpy cooling counters to device");
    copy_to_device(
        move_counters, host_moves, "cudaMemcpy move counters to device");
    copy_to_device(
        candidate_counters, host_candidate_evaluations,
        "cudaMemcpy candidate counters to device");

    const int initial_best_replica =
        select_best_replica(host_best_scores, host_best_conflicts);
    const int initially_solved =
        host_best_scores[static_cast<size_t>(initial_best_replica)] == 0
            ? 1
            : 0;
    check_cuda(
        cudaMemcpy(
            solved.data(), &initially_solved, sizeof(initially_solved),
            cudaMemcpyHostToDevice),
        "cudaMemcpy solved flag to device");

    const int rng_blocks = static_cast<int>(
        (static_cast<long long>(replicas) + k_block_threads - 1) /
        k_block_threads);
    const unsigned long long anneal_seed =
        (static_cast<unsigned long long>(base_seed) << 32) ^
        static_cast<unsigned long long>(params.seed) ^
        UINT64_C(0xd1b54a32d192ed03);
    initialize_rng_kernel<<<rng_blocks, k_block_threads>>>(
        rng_states.data(), replicas, anneal_seed);
    check_cuda(cudaGetLastError(), "launch initialize_rng_kernel");
    check_cuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize initialize_rng_kernel");

    DeviceStorage storage;
    storage.current_entries = current_entries.data();
    storage.current_gram = current_gram.data();
    storage.current_residual = current_residual.data();
    storage.current_minority = current_minority.data();
    storage.current_score = current_scores.data();
    storage.current_conflicts = current_conflicts.data();
    storage.best_entries = best_entries.data();
    storage.best_score = best_scores.data();
    storage.best_conflicts = best_conflicts.data();
    storage.temperature = temperatures.data();
    storage.reheat_temperature = reheat_temperatures.data();
    storage.moves_since_improvement = moves_since_improvement.data();
    storage.cooling_counter = cooling_counters.data();
    storage.rng = rng_states.data();
    storage.moves = move_counters.data();
    storage.candidate_evaluations = candidate_counters.data();
    storage.solved = solved.data();

    KernelParams kernel_params;
    kernel_params.order = order;
    kernel_params.weight = weight;
    kernel_params.minority_count = minority_count;
    kernel_params.sparse_side = sparse_side ? 1 : 0;
    kernel_params.moves_per_launch = params.moves_per_launch;
    kernel_params.candidate_samples = params.candidate_samples;
    kernel_params.target_samples = params.target_samples;
    kernel_params.moves_per_cool = params.moves_per_cool;
    kernel_params.stuck_threshold = params.stuck_threshold;
    kernel_params.sign_fraction = params.sign_fraction;
    kernel_params.double_sign_fraction =
        params.double_sign_fraction;
    kernel_params.switch_sign_fraction =
        params.switch_sign_fraction;
    kernel_params.squared_objective_fraction =
        params.squared_objective_fraction;
    kernel_params.parity_objective_fraction =
        params.parity_objective_fraction;
    kernel_params.greedy_fraction = params.greedy_fraction;
    kernel_params.target_fraction = params.target_fraction;
    kernel_params.minimum_temperature = params.t_min;
    kernel_params.cooling = params.cooling;
    kernel_params.reheat = params.reheat;

    const double initialization_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - initialization_start).count();
    std::cout << "[init] seconds=" << initialization_seconds
              << " replica_mib="
              << static_cast<double>(replica_bytes) /
                     (1024.0 * 1024.0)
              << " total_replica_mib="
              << static_cast<double>(replica_bytes) * replicas /
                     (1024.0 * 1024.0)
              << '\n';
    std::cout << "[t=0s] start score="
              << host_best_scores[
                     static_cast<size_t>(initial_best_replica)]
              << " conflicts="
              << host_best_conflicts[
                     static_cast<size_t>(initial_best_replica)]
              << " n=" << order
              << " w=" << weight
              << " replicas=" << replicas
              << " t_init=" << initial_temperature
              << '\n';

    int best_replica = initial_best_replica;
    long long global_best_score =
        host_best_scores[static_cast<size_t>(best_replica)];
    long long global_best_conflicts =
        host_best_conflicts[static_cast<size_t>(best_replica)];
    bool checkpoint_pending = true;
    double last_save_time = -std::numeric_limits<double>::infinity();
    double last_progress_time = 0.0;
    double last_improvement_time = 0.0;
    double last_restart_time = 0.0;
    size_t restart_cursor = 0;
    uint64_t restart_round = 0;
    uint64_t restarted_replicas = 0;
    const auto search_start = std::chrono::steady_clock::now();

    auto restart_population =
        [&](int elite_replica, double elapsed) {
            if (params.restart_interval <= 0.0 || replicas < 2)
                return 0;
            const int available = replicas - 1;
            const int restart_count = std::clamp(
                static_cast<int>(std::ceil(
                    params.restart_fraction * available)),
                1, available);

            copy_from_device(
                host_entries, current_entries,
                "cudaMemcpy restart current entries to host");
            copy_from_device(
                host_gram, current_gram,
                "cudaMemcpy restart current Gram to host");
            copy_from_device(
                host_residual, current_residual,
                "cudaMemcpy restart current residual to host");
            copy_from_device(
                host_minority, current_minority,
                "cudaMemcpy restart current minority to host");
            copy_from_device(
                host_current_scores, current_scores,
                "cudaMemcpy restart current scores to host");
            copy_from_device(
                host_current_conflicts, current_conflicts,
                "cudaMemcpy restart current conflicts to host");
            copy_from_device(
                host_best_entries, best_entries,
                "cudaMemcpy restart best entries to host");
            copy_from_device(
                host_temperatures, temperatures,
                "cudaMemcpy restart temperatures to host");
            copy_from_device(
                host_moves_since_improvement, moves_since_improvement,
                "cudaMemcpy restart counters to host");
            copy_from_device(
                host_cooling_counters, cooling_counters,
                "cudaMemcpy restart cooling counters to host");

            const size_t elite_base =
                static_cast<size_t>(elite_replica) * matrix_items;
            std::vector<int8_t> elite_entries(matrix_items);
            std::copy_n(
                host_best_entries.begin() +
                    static_cast<std::ptrdiff_t>(elite_base),
                static_cast<std::ptrdiff_t>(matrix_items),
                elite_entries.begin());
            const xweigh::State elite =
                xweigh::State::from_entries(
                    order, weight, elite_entries);

            struct EscapeMove {
                xweigh::Move support;
                bool flip_first = false;
                bool flip_second = false;
                int64_t score_delta = 0;
                int64_t conflict_delta = 0;
            };
            const size_t escape_limit =
                std::max<size_t>(
                    static_cast<size_t>(restart_count) * 4, 1);
            const auto escape_less =
                [](const EscapeMove& first,
                   const EscapeMove& second) {
                    if (first.score_delta != second.score_delta)
                        return first.score_delta < second.score_delta;
                    return first.conflict_delta <
                           second.conflict_delta;
                };
            std::vector<int> elite_gram(matrix_items, 0);
            for (int first = 0; first < order; ++first) {
                for (int second = first + 1;
                     second < order; ++second) {
                    int dot = 0;
                    for (int column = 0;
                         column < order; ++column) {
                        dot +=
                            elite_entries[
                                static_cast<size_t>(column) *
                                    order + first] *
                            elite_entries[
                                static_cast<size_t>(column) *
                                    order + second];
                    }
                    elite_gram[
                        static_cast<size_t>(first) *
                            order + second] = dot;
                    elite_gram[
                        static_cast<size_t>(second) *
                            order + first] = dot;
                }
            }
            std::vector<int> minority_position(matrix_items, -1);
            for (int row = 0; row < order; ++row) {
                int position = 0;
                for (int column = 0;
                     column < order; ++column) {
                    const bool nonzero =
                        elite_entries[
                            static_cast<size_t>(column) *
                                order + row] != 0;
                    if (nonzero == sparse_side) {
                        minority_position[
                            static_cast<size_t>(row) *
                                order + column] = position++;
                    }
                }
            }

            std::vector<EscapeMove> escape_moves;
            for (int row1 = 0; row1 < order; ++row1) {
                for (int row2 = row1 + 1;
                     row2 < order; ++row2) {
                    for (int col1 = 0; col1 < order; ++col1) {
                        const int old_value1 =
                            elite_entries[
                                static_cast<size_t>(col1) *
                                    order + row1];
                        if (old_value1 == 0 ||
                            elite_entries[
                                static_cast<size_t>(col1) *
                                    order + row2] != 0) {
                            continue;
                        }
                        for (int col2 = 0;
                             col2 < order; ++col2) {
                            const int old_value2 =
                                elite_entries[
                                    static_cast<size_t>(col2) *
                                        order + row2];
                            if (old_value2 == 0 ||
                                elite_entries[
                                    static_cast<size_t>(col2) *
                                        order + row1] != 0) {
                                continue;
                            }
                            for (int new_value1 : { -1, 1 }) {
                                for (int new_value2 : { -1, 1 }) {
                                    int64_t score_delta = 0;
                                    int64_t conflict_delta = 0;
                                    for (int other = 0;
                                         other < order; ++other) {
                                        if (other == row1 ||
                                            other == row2) {
                                            continue;
                                        }
                                        const int old1 =
                                            elite_gram[
                                                static_cast<size_t>(
                                                    row1) *
                                                    order + other];
                                        const int old2 =
                                            elite_gram[
                                                static_cast<size_t>(
                                                    row2) *
                                                    order + other];
                                        const int new1 = old1 +
                                            new_value1 *
                                                elite_entries[
                                                    static_cast<size_t>(
                                                        col2) *
                                                        order + other] -
                                            old_value1 *
                                                elite_entries[
                                                    static_cast<size_t>(
                                                        col1) *
                                                        order + other];
                                        const int new2 = old2 +
                                            new_value2 *
                                                elite_entries[
                                                    static_cast<size_t>(
                                                        col1) *
                                                        order + other] -
                                            old_value2 *
                                                elite_entries[
                                                    static_cast<size_t>(
                                                        col2) *
                                                        order + other];
                                        score_delta +=
                                            std::abs(new1) -
                                            std::abs(old1);
                                        score_delta +=
                                            std::abs(new2) -
                                            std::abs(old2);
                                        conflict_delta +=
                                            (new1 != 0) -
                                            (old1 != 0);
                                        conflict_delta +=
                                            (new2 != 0) -
                                            (old2 != 0);
                                    }
                                    xweigh::Move support;
                                    support.kind =
                                        xweigh::MoveKind::support_switch;
                                    support.row1 = row1;
                                    support.row2 = row2;
                                    support.col1 = col1;
                                    support.col2 = col2;
                                    support.minority_pos1 =
                                        minority_position[
                                            static_cast<size_t>(row1) *
                                                order +
                                            (sparse_side
                                                ? col1 : col2)];
                                    support.minority_pos2 =
                                        minority_position[
                                            static_cast<size_t>(row2) *
                                                order +
                                            (sparse_side
                                                ? col2 : col1)];
                                    escape_moves.push_back({
                                        support,
                                        new_value1 != old_value1,
                                        new_value2 != old_value2,
                                        score_delta,
                                        conflict_delta
                                    });
                                    if (escape_moves.size() >
                                        escape_limit * 2) {
                                        std::nth_element(
                                            escape_moves.begin(),
                                            escape_moves.begin() +
                                                static_cast<
                                                    std::ptrdiff_t>(
                                                    escape_limit),
                                            escape_moves.end(),
                                            escape_less);
                                        escape_moves.resize(
                                            escape_limit);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            std::sort(
                escape_moves.begin(), escape_moves.end(),
                escape_less);
            if (escape_moves.size() > escape_limit)
                escape_moves.resize(escape_limit);

            std::vector<int> selected;
            selected.reserve(static_cast<size_t>(restart_count));
            size_t scanned = 0;
            while (selected.size() <
                       static_cast<size_t>(restart_count) &&
                   scanned < static_cast<size_t>(replicas) * 2) {
                const int replica = static_cast<int>(
                    (restart_cursor + scanned) %
                    static_cast<size_t>(replicas));
                ++scanned;
                if (replica == elite_replica) continue;
                selected.push_back(replica);
            }
            restart_cursor =
                (restart_cursor + scanned) %
                static_cast<size_t>(replicas);

            int minimum_kick = params.restart_kick_min;
            int maximum_kick = params.restart_kick_max;
            for (size_t index = 0; index < selected.size(); ++index) {
                const int replica = selected[index];
                const int kick_steps = selected.size() > 1
                    ? minimum_kick +
                        static_cast<int>(
                            index * static_cast<size_t>(
                                maximum_kick - minimum_kick) /
                            (selected.size() - 1))
                    : minimum_kick;
                xweigh::State kicked = elite;
                std::mt19937 rng(
                    base_seed ^
                    static_cast<uint32_t>(
                        restart_round * UINT64_C(0x9e3779b97f4a7c15)) ^
                    (static_cast<uint32_t>(replica) *
                     UINT32_C(0x85ebca6b)));
                if (!escape_moves.empty()) {
                    const size_t pool_size = std::min(
                        escape_moves.size(),
                        std::max<size_t>(
                            selected.size() * 4, 1));
                    const size_t escape_index =
                        (index +
                         static_cast<size_t>(restart_round) *
                             selected.size()) %
                        pool_size;
                    const EscapeMove& escape =
                        escape_moves[escape_index];
                    xweigh::Evaluation evaluation =
                        kicked.evaluate(escape.support);
                    kicked.commit(escape.support, evaluation);
                    if (escape.flip_first) {
                        xweigh::Move flip;
                        flip.kind = xweigh::MoveKind::sign_flip;
                        flip.row1 = escape.support.row1;
                        flip.col1 = escape.support.col2;
                        evaluation = kicked.evaluate(flip);
                        kicked.commit(flip, evaluation);
                    }
                    if (escape.flip_second) {
                        xweigh::Move flip;
                        flip.kind = xweigh::MoveKind::sign_flip;
                        flip.row1 = escape.support.row2;
                        flip.col1 = escape.support.col1;
                        evaluation = kicked.evaluate(flip);
                        kicked.commit(flip, evaluation);
                    }
                }
                const int kick_mode =
                    static_cast<int>((index + restart_round) % 3);
                int random_steps = kick_steps;
                if (kick_mode == 0) {
                    random_steps = 0;
                } else if (kick_mode == 1 &&
                           kick_steps > 0 && weight >= 3) {
                    const int row_kicks = std::clamp(
                        1 + kick_steps / std::max(1, weight),
                        1, 4);
                    const int maximum_flips =
                        std::max(2, weight / 2);
                    for (int row_kick = 0;
                         row_kick < row_kicks; ++row_kick) {
                        const int row =
                            std::uniform_int_distribution<int>(
                                0, order - 1)(rng);
                        std::vector<int> nonzero_columns;
                        nonzero_columns.reserve(
                            static_cast<size_t>(weight));
                        for (int column = 0;
                             column < order; ++column) {
                            if (kicked.entries()[
                                    static_cast<size_t>(column) *
                                        order + row] != 0) {
                                nonzero_columns.push_back(column);
                            }
                        }
                        std::shuffle(
                            nonzero_columns.begin(),
                            nonzero_columns.end(), rng);
                        const int flip_count =
                            std::uniform_int_distribution<int>(
                                2, maximum_flips)(rng);
                        for (int flip = 0;
                             flip < flip_count; ++flip) {
                            xweigh::Move move;
                            move.kind =
                                xweigh::MoveKind::sign_flip;
                            move.row1 = row;
                            move.col1 =
                                nonzero_columns[
                                    static_cast<size_t>(flip)];
                            const xweigh::Evaluation evaluation =
                                kicked.evaluate(move);
                            kicked.commit(move, evaluation);
                        }
                    }
                    random_steps = kick_steps / 4;
                }
                xweigh::Params kick_params = cpu_params;
                if (kick_mode == 2) kick_params.sign_fraction = 0.0;
                int random_applied = 0;
                for (int attempt = 0;
                     attempt < random_steps * 8 &&
                     random_applied < random_steps; ++attempt) {
                    xweigh::Move move;
                    if (!kicked.propose_move(
                            rng, kick_params, false, move)) {
                        continue;
                    }
                    const xweigh::Evaluation evaluation =
                        kicked.evaluate(move);
                    kicked.commit(move, evaluation);
                    ++random_applied;
                }

                const size_t residual_base =
                    static_cast<size_t>(replica) * n;
                std::fill_n(
                    host_residual.begin() +
                        static_cast<std::ptrdiff_t>(residual_base),
                    static_cast<std::ptrdiff_t>(n), 0);
                const PackedScore packed = pack_state(
                    kicked, replica, order, weight,
                    minority_count, sparse_side, host_entries,
                    host_gram, host_residual, host_minority);
                host_current_scores[static_cast<size_t>(replica)] =
                    packed.score;
                host_current_conflicts[static_cast<size_t>(replica)] =
                    packed.conflicts;

                const bool kick_is_better = xweigh::better_pair(
                    packed.score, packed.conflicts,
                    global_best_score, global_best_conflicts);
                const auto& replica_best =
                    kick_is_better ? kicked.entries() : elite_entries;
                const size_t matrix_base =
                    static_cast<size_t>(replica) * matrix_items;
                std::copy(
                    replica_best.begin(), replica_best.end(),
                    host_best_entries.begin() +
                        static_cast<std::ptrdiff_t>(matrix_base));
                host_best_scores[static_cast<size_t>(replica)] =
                    kick_is_better
                        ? packed.score : global_best_score;
                host_best_conflicts[static_cast<size_t>(replica)] =
                    kick_is_better
                        ? packed.conflicts : global_best_conflicts;
                host_temperatures[static_cast<size_t>(replica)] =
                    std::max(
                        params.t_min,
                        host_reheat_temperatures[
                            static_cast<size_t>(replica)] *
                            params.reheat);
                host_moves_since_improvement[
                    static_cast<size_t>(replica)] = 0;
                host_cooling_counters[
                    static_cast<size_t>(replica)] = 0;
            }

            copy_to_device(
                current_entries, host_entries,
                "cudaMemcpy restart current entries to device");
            copy_to_device(
                current_gram, host_gram,
                "cudaMemcpy restart current Gram to device");
            copy_to_device(
                current_residual, host_residual,
                "cudaMemcpy restart current residual to device");
            copy_to_device(
                current_minority, host_minority,
                "cudaMemcpy restart current minority to device");
            copy_to_device(
                current_scores, host_current_scores,
                "cudaMemcpy restart current scores to device");
            copy_to_device(
                current_conflicts, host_current_conflicts,
                "cudaMemcpy restart current conflicts to device");
            copy_to_device(
                best_entries, host_best_entries,
                "cudaMemcpy restart best entries to device");
            copy_to_device(
                best_scores, host_best_scores,
                "cudaMemcpy restart best scores to device");
            copy_to_device(
                best_conflicts, host_best_conflicts,
                "cudaMemcpy restart best conflicts to device");
            copy_to_device(
                temperatures, host_temperatures,
                "cudaMemcpy restart temperatures to device");
            copy_to_device(
                moves_since_improvement,
                host_moves_since_improvement,
                "cudaMemcpy restart counters to device");
            copy_to_device(
                cooling_counters, host_cooling_counters,
                "cudaMemcpy restart cooling counters to device");

            ++restart_round;
            restarted_replicas += selected.size();
            std::cout << "[restart] seconds=" << elapsed
                      << " elite_score=" << global_best_score
                      << " replicas=" << selected.size()
                      << " kicks=" << minimum_kick << ".."
                      << maximum_kick
                      << " min_escape_delta="
                      << (escape_moves.empty()
                              ? 0 : escape_moves.front().score_delta)
                      << '\n';
            return static_cast<int>(selected.size());
        };

    while (global_best_score != 0) {
        const double before_launch = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - search_start).count();
        if (g_stop.load(std::memory_order_relaxed) ||
            (params.max_seconds > 0.0 &&
             before_launch >= params.max_seconds)) {
            break;
        }

        anneal_kernel<<<replicas, k_block_threads, shared_layout.bytes>>>(
            storage, kernel_params);
        check_cuda(cudaGetLastError(), "launch anneal_kernel");
        check_cuda(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize anneal_kernel");

        copy_from_device(
            host_best_scores, best_scores,
            "cudaMemcpy best scores to host");
        copy_from_device(
            host_best_conflicts, best_conflicts,
            "cudaMemcpy best conflicts to host");
        copy_from_device(
            host_moves, move_counters,
            "cudaMemcpy move counters to host");
        copy_from_device(
            host_candidate_evaluations, candidate_counters,
            "cudaMemcpy candidate counters to host");

        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - search_start).count();
        const int candidate_best =
            select_best_replica(host_best_scores, host_best_conflicts);
        if (xweigh::better_pair(
                host_best_scores[static_cast<size_t>(candidate_best)],
                host_best_conflicts[static_cast<size_t>(candidate_best)],
                global_best_score, global_best_conflicts)) {
            best_replica = candidate_best;
            global_best_score =
                host_best_scores[static_cast<size_t>(best_replica)];
            global_best_conflicts =
                host_best_conflicts[static_cast<size_t>(best_replica)];
            checkpoint_pending = true;
            last_improvement_time = elapsed;
            std::cout << "[t=" << elapsed << "s] new best score="
                      << global_best_score
                      << " conflicts=" << global_best_conflicts
                      << " (replica " << best_replica << ")\n";
        }

        if (params.restart_interval > 0.0 &&
            elapsed - last_improvement_time >=
                params.restart_interval &&
            elapsed - last_restart_time >=
                params.restart_interval) {
            restart_population(best_replica, elapsed);
            last_restart_time = elapsed;
        }

        if (checkpoint_pending &&
            elapsed - last_save_time >= params.save_interval) {
            try {
                const std::vector<int8_t> snapshot =
                    copy_best_entries(
                        best_entries, best_replica, matrix_items);
                xweigh::write_entries_csv(snapshot, order, params.output);
                checkpoint_pending = false;
            } catch (const std::exception& error) {
                std::cerr << "Warning: " << error.what() << '\n';
            }
            last_save_time = elapsed;
        }

        if (elapsed - last_progress_time >= 2.0 &&
            global_best_score != 0) {
            std::cout << "[t=" << elapsed << "s] moves="
                      << sum_counters(host_moves)
                      << " best score=" << global_best_score
                      << " conflicts=" << global_best_conflicts
                      << '\n';
            last_progress_time = elapsed;
        }
    }

    copy_from_device(
        host_best_scores, best_scores,
        "cudaMemcpy final best scores to host");
    copy_from_device(
        host_best_conflicts, best_conflicts,
        "cudaMemcpy final best conflicts to host");
    copy_from_device(
        host_moves, move_counters,
        "cudaMemcpy final move counters to host");
    copy_from_device(
        host_candidate_evaluations, candidate_counters,
        "cudaMemcpy final candidate counters to host");
    best_replica = select_best_replica(
        host_best_scores, host_best_conflicts);
    global_best_score =
        host_best_scores[static_cast<size_t>(best_replica)];
    global_best_conflicts =
        host_best_conflicts[static_cast<size_t>(best_replica)];

    std::vector<int8_t> final_entries =
        copy_best_entries(best_entries, best_replica, matrix_items);
    xweigh::State best = xweigh::State::from_entries(
        order, weight, final_entries);
    if (best.score() != global_best_score ||
        best.conflicts() != global_best_conflicts) {
        throw std::runtime_error(
            "CUDA final best snapshot score mismatch");
    }
    xweigh::write_entries_csv(final_entries, order, params.output);

    const double search_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - search_start).count();
    const uint64_t total_moves = sum_counters(host_moves);
    const uint64_t total_candidate_evaluations =
        sum_counters(host_candidate_evaluations);
    std::cout << "done moves=" << total_moves
              << " candidate_evaluations="
              << total_candidate_evaluations
              << " seconds=" << search_seconds
              << " moves_per_sec="
              << (search_seconds > 0.0
                      ? static_cast<double>(total_moves) / search_seconds
                      : 0.0)
              << " candidate_evaluations_per_sec="
              << (search_seconds > 0.0
                      ? static_cast<double>(total_candidate_evaluations) /
                            search_seconds
                      : 0.0)
              << " restarted_replicas=" << restarted_replicas
              << '\n';

    RunResult result;
    result.best = std::move(best);
    result.device = device;
    result.replicas = replicas;
    result.moves = total_moves;
    result.candidate_evaluations = total_candidate_evaluations;
    result.restarted_replicas = restarted_replicas;
    result.initialization_seconds = initialization_seconds;
    result.search_seconds = search_seconds;
    return result;
}

}  // namespace xweigh_cuda
