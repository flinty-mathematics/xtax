#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "xweigh_anneal.hpp"

namespace xweigh_cuda {

constexpr int MAX_CANDIDATE_SAMPLES = 32;
constexpr int MAX_TARGET_SAMPLES = 64;

struct Params {
    int device = 0;
    int replicas = 0;
    int moves_per_launch = 4096;
    uint64_t seed = 0;
    double max_seconds = 0.0;
    double save_interval = 2.0;
    std::string output = "best_W.csv";

    double sign_fraction = 0.5;
    double double_sign_fraction = 0.25;
    double switch_sign_fraction = 0.5;
    double squared_objective_fraction = 0.25;
    double parity_objective_fraction = 0.25;
    double greedy_fraction = 0.5;
    int candidate_samples = 4;
    double target_fraction = 0.7;
    int target_samples = 8;

    double t_init = 0.0;
    double t_min = 0.25;
    double cooling = 0.999;
    int moves_per_cool = 500;
    int stuck_threshold = 50000;
    double reheat = 1.0;

    double restart_interval = 5.0;
    double restart_fraction = 0.25;
    int restart_kick_min = 2;
    int restart_kick_max = 64;
};

struct DeviceInfo {
    std::string name;
    int ordinal = 0;
    int compute_major = 0;
    int compute_minor = 0;
    int multiprocessors = 0;
    size_t global_memory = 0;
    size_t max_shared_memory_per_block = 0;
};

struct RunResult {
    xweigh::State best;
    DeviceInfo device;
    int replicas = 0;
    uint64_t moves = 0;
    uint64_t candidate_evaluations = 0;
    uint64_t restarted_replicas = 0;
    double initialization_seconds = 0.0;
    double search_seconds = 0.0;
};

DeviceInfo query_device(int ordinal);

RunResult run_annealer(
    int order, int weight, const Params& params,
    const xweigh::State* initial_state = nullptr);

}  // namespace xweigh_cuda
