// Win32 dashboard for xbkz: per-worker progress bars, phase labels, global best.

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commctrl.h>

#include "xbkz_ui.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#pragma comment(lib, "comctl32.lib")
// Themed common controls (v6) so marquee progress bars render correctly.
#pragma comment(linker, \
    "\"/manifestdependency:type='win32' " \
    "name='Microsoft.Windows.Common-Controls' version='6.0.0.0' " \
    "processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

namespace {

constexpr int k_timer_ms = 100;
constexpr UINT_PTR k_timer_id = 1;
constexpr int k_pb_max = 1000;
constexpr int k_marquee_ms = 30;

constexpr int IDC_BEST_TEXT = 101;
constexpr int IDC_ELAPSED_TEXT = 102;
constexpr int IDC_IMPROV_TEXT = 103;
constexpr int IDC_PARAMS_TEXT = 104;
constexpr int IDC_WORKER_BASE = 200;

constexpr int k_margin = 14;
constexpr int k_row = 26;
constexpr int k_pb_h = 18;
constexpr int k_header_rows = 3;

struct UiState {
    HWND hwnd = nullptr;
    HWND hwnd_best = nullptr;
    HWND hwnd_elapsed = nullptr;
    HWND hwnd_improv = nullptr;
    HWND hwnd_params = nullptr;
    std::vector<HWND> worker_id;
    std::vector<HWND> worker_pb;
    std::vector<HWND> worker_phase;
    HFONT font = nullptr;
    HFONT font_title = nullptr;
    const BkzUiConfig* cfg = nullptr;
    SteadyTimePoint start{};
    std::vector<WorkerStatus>* workers = nullptr;
    const GlobalBest* best = nullptr;
    std::atomic<bool>* stop_flag = nullptr;
    std::function<bool()> finished;
    std::vector<bool> worker_marquee;
    std::vector<int> worker_pb_state;
};

static bool phase_is_indeterminate(WorkerPhase p) {
    return p == WorkerPhase::starting || p == WorkerPhase::init_lll
           || p == WorkerPhase::reseed;
}

// Themed progress bars cannot take an arbitrary fill colour, but PBM_SETSTATE
// switches between built-in palettes: green (normal) while a block is being
// enumerated, yellow (paused) while a block is being sieved, so the two oracles
// are visually distinct as the worker moves through the tour.
static int phase_bar_state(WorkerPhase p) {
    return (p == WorkerPhase::sieving) ? PBST_PAUSED : PBST_NORMAL;
}

// PBS_MARQUEE and PBS_SMOOTH are mutually exclusive; marquee must be toggled with
// a frame refresh or the stripe never appears.
static void set_progress_marquee(HWND pb, bool on) {
    if (!pb) return;

    if (on) {
        LONG style = GetWindowLongW(pb, GWL_STYLE);
        style &= ~PBS_SMOOTH;
        style |= PBS_MARQUEE;
        SetWindowLongW(pb, GWL_STYLE, style);
        SetWindowPos(pb, nullptr, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
                         | SWP_FRAMECHANGED);
        SendMessageW(pb, PBM_SETMARQUEE, TRUE, (LPARAM)k_marquee_ms);
    } else {
        SendMessageW(pb, PBM_SETMARQUEE, FALSE, 0);
        LONG style = GetWindowLongW(pb, GWL_STYLE);
        style &= ~PBS_MARQUEE;
        SetWindowLongW(pb, GWL_STYLE, style);
        SetWindowPos(pb, nullptr, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
                         | SWP_FRAMECHANGED);
        SendMessageW(pb, PBM_SETRANGE32, 0, k_pb_max);
        SendMessageW(pb, PBM_SETPOS, 0, 0);
    }
    InvalidateRect(pb, nullptr, TRUE);
}

static double elapsed_s(SteadyTimePoint start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

static std::wstring fmt_double(double v, int prec = 6) {
    wchar_t buf[64];
    std::swprintf(buf, 64, L"%.*g", prec, v);
    return buf;
}

static std::wstring fmt_count(long long v) {
    wchar_t buf[32];
    if (v < 1000)
        std::swprintf(buf, 32, L"%lld", v);
    else if (v < 1000000)
        std::swprintf(buf, 32, L"%.1fk", (double)v / 1e3);
    else
        std::swprintf(buf, 32, L"%.1fM", (double)v / 1e6);
    return buf;
}

static HFONT make_font(int height, bool bold) {
    return CreateFontW(
        height, 0, 0, 0, bold ? FW_SEMIBOLD : FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
}

static void set_font(HWND hwnd, HFONT font) {
    if (hwnd && font) SendMessageW(hwnd, WM_SETFONT, (WPARAM)font, TRUE);
}

static int worker_progress(const WorkerStatus& w) {
    const auto phase = (WorkerPhase)w.phase.load(std::memory_order_relaxed);
    if (phase_is_indeterminate(phase))
        return 0;

    switch (phase) {
    // Both oracles advance block by block through the tour, so the bar tracks
    // the same block fraction in either phase and rises continuously across a
    // tour even as blocks switch between enumeration and sieving.
    case WorkerPhase::tour:
    case WorkerPhase::sieving: {
        const int bi = w.block_idx.load(std::memory_order_relaxed);
        const int bt = w.block_total.load(std::memory_order_relaxed);
        if (bt <= 0) return 0;
        int pos = (int)std::lround(1000.0 * (double)bi / (double)bt);
        if (pos < 0) pos = 0;
        if (pos > k_pb_max) pos = k_pb_max;
        return pos;
    }
    case WorkerPhase::stopped:
        return k_pb_max;
    default:
        return 0;
    }
}

static std::wstring worker_phase_text(const WorkerStatus& w) {
    const auto phase = (WorkerPhase)w.phase.load(std::memory_order_relaxed);
    const int tour = w.tour.load(std::memory_order_relaxed);
    const int cur = w.cur_beta.load(std::memory_order_relaxed);
    const int tgt = w.target_beta.load(std::memory_order_relaxed);
    const int bi = w.block_idx.load(std::memory_order_relaxed);
    const int bt = w.block_total.load(std::memory_order_relaxed);

    wchar_t buf[256];
    switch (phase) {
    case WorkerPhase::starting:
        return L"Starting";
    case WorkerPhase::init_lll:
        return L"Initial LLL reduction";
    case WorkerPhase::warmup:
        std::swprintf(buf, 256, L"Warmup  \u03B2=%d/%d  tour %d", cur, tgt, tour);
        return buf;
    case WorkerPhase::tour:
        std::swprintf(buf, 256, L"Tour  \u03B2=%d  block %d/%d  %s nodes", cur, bi, bt,
                      fmt_count(w.enum_nodes.load(std::memory_order_relaxed)).c_str());
        return buf;
    case WorkerPhase::sieving:
        std::swprintf(buf, 256, L"Sieving  \u03B2=%d  block %d/%d", cur, bi, bt);
        return buf;
    case WorkerPhase::reseed:
        return L"Reseeding from global best";
    case WorkerPhase::stopped:
        return L"Stopped";
    }
    return L"?";
}

static std::wstring worker_phase_label(const WorkerStatus& w, bool shutting_down) {
    const auto phase = (WorkerPhase)w.phase.load(std::memory_order_relaxed);
    if (shutting_down && phase != WorkerPhase::stopped)
        return L"Stopping";
    return worker_phase_text(w);
}

static void layout_client(UiState* ui) {
    if (!ui || !ui->hwnd) return;
    RECT rc{};
    GetClientRect(ui->hwnd, &rc);
    const int W = rc.right - rc.left;
    const int H = rc.bottom - rc.top;
    if (W <= 0 || H <= 0) return;

    int y = k_margin;
    const int id_w = 52;
    const int phase_w = 320;
    const int pb_x = k_margin + id_w + 8;
    const int pb_w = W - pb_x - phase_w - k_margin - 8;
    const int phase_x = pb_x + pb_w + 8;

    auto place = [&](HWND h, int x, int yy, int w, int hgt) {
        if (h) SetWindowPos(h, nullptr, x, yy, w, hgt, SWP_NOZORDER | SWP_NOACTIVATE);
    };

    place(ui->hwnd_best, k_margin, y, W - 2 * k_margin, k_row);
    y += k_row;
    place(ui->hwnd_elapsed, k_margin, y, (W - 2 * k_margin) / 2, k_row);
    place(ui->hwnd_improv, k_margin + (W - 2 * k_margin) / 2, y,
          (W - 2 * k_margin) - (W - 2 * k_margin) / 2, k_row);
    y += k_row;
    place(ui->hwnd_params, k_margin, y, W - 2 * k_margin, k_row);
    y += k_row + 8;

    for (size_t i = 0; i < ui->worker_pb.size(); ++i) {
        place(ui->worker_id[i], k_margin, y, id_w, k_row);
        place(ui->worker_pb[i], pb_x, y + 3, pb_w, k_pb_h);
        place(ui->worker_phase[i], phase_x, y, phase_w, k_row);
        y += k_row + 4;
    }
    (void)H;
}

static void refresh_ui(UiState* ui) {
    if (!ui) return;

    wchar_t buf[512];
    if (ui->best->has_relaxed.load(std::memory_order_acquire)) {
        const double b0 = ui->best->b0_relaxed.load(std::memory_order_relaxed);
        const double norm = std::sqrt(b0);
        std::swprintf(buf, 512, L"Best shortest norm: %s   (norm\u00B2 = %s)",
                      fmt_double(norm, 8).c_str(), fmt_double(b0, 6).c_str());
    } else {
        std::swprintf(buf, 512, L"Best shortest norm: \u2014");
    }
    SetWindowTextW(ui->hwnd_best, buf);

    const double elapsed = elapsed_s(ui->start);
    std::swprintf(buf, 512, L"Elapsed: %.1f s", elapsed);
    SetWindowTextW(ui->hwnd_elapsed, buf);

    std::swprintf(buf, 512, L"Improvements: %lld",
                  ui->best->improvements.load(std::memory_order_relaxed));
    SetWindowTextW(ui->hwnd_improv, buf);

    std::swprintf(buf, 512,
                  L"%d workers  |  block range %d..%d  |  lattice %d\u00D7%d",
                  ui->cfg->threads, ui->cfg->block_start, ui->cfg->block,
                  ui->cfg->lattice_m, ui->cfg->lattice_d);
    SetWindowTextW(ui->hwnd_params, buf);

    for (size_t i = 0; i < ui->workers->size(); ++i) {
        const WorkerStatus& w = (*ui->workers)[i];
        const auto phase = (WorkerPhase)w.phase.load(std::memory_order_relaxed);
        const bool indet = phase_is_indeterminate(phase);

        if (indet != ui->worker_marquee[i]) {
            set_progress_marquee(ui->worker_pb[i], indet);
            ui->worker_marquee[i] = indet;
        }
        const int state = phase_bar_state(phase);
        if (state != ui->worker_pb_state[i]) {
            SendMessageW(ui->worker_pb[i], PBM_SETSTATE, (WPARAM)state, 0);
            ui->worker_pb_state[i] = state;
        }
        if (!indet) {
            const int pos = worker_progress(w);
            SendMessageW(ui->worker_pb[i], PBM_SETPOS, (WPARAM)pos, 0);
        }
        const bool shutting_down =
            ui->stop_flag && ui->stop_flag->load(std::memory_order_relaxed);
        SetWindowTextW(ui->worker_phase[i],
                       worker_phase_label(w, shutting_down).c_str());
    }
}

static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    UiState* ui = (UiState*)GetWindowLongPtrW(hwnd, GWLP_USERDATA);

    switch (msg) {
    case WM_CREATE: {
        auto* cs = (CREATESTRUCTW*)lp;
        ui = (UiState*)cs->lpCreateParams;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)ui);
        ui->hwnd = hwnd;
        ui->font = make_font(-15, false);
        ui->font_title = make_font(-17, true);

        ui->hwnd_best = CreateWindowExW(
            0, L"STATIC", L"", WS_CHILD | WS_VISIBLE | SS_LEFTNOWORDWRAP,
            0, 0, 100, k_row, hwnd, (HMENU)(INT_PTR)IDC_BEST_TEXT, nullptr, nullptr);
        ui->hwnd_elapsed = CreateWindowExW(
            0, L"STATIC", L"", WS_CHILD | WS_VISIBLE,
            0, 0, 100, k_row, hwnd, (HMENU)(INT_PTR)IDC_ELAPSED_TEXT, nullptr, nullptr);
        ui->hwnd_improv = CreateWindowExW(
            0, L"STATIC", L"", WS_CHILD | WS_VISIBLE,
            0, 0, 100, k_row, hwnd, (HMENU)(INT_PTR)IDC_IMPROV_TEXT, nullptr, nullptr);
        ui->hwnd_params = CreateWindowExW(
            0, L"STATIC", L"", WS_CHILD | WS_VISIBLE,
            0, 0, 100, k_row, hwnd, (HMENU)(INT_PTR)IDC_PARAMS_TEXT, nullptr, nullptr);

        set_font(ui->hwnd_best, ui->font_title);
        set_font(ui->hwnd_elapsed, ui->font);
        set_font(ui->hwnd_improv, ui->font);
        set_font(ui->hwnd_params, ui->font);

        const int n = ui->cfg ? ui->cfg->threads : 0;
        ui->worker_id.resize((size_t)n);
        ui->worker_pb.resize((size_t)n);
        ui->worker_phase.resize((size_t)n);
        ui->worker_marquee.assign((size_t)n, false);
        ui->worker_pb_state.assign((size_t)n, -1);
        for (int i = 0; i < n; ++i) {
            const int base = IDC_WORKER_BASE + i * 3;
            wchar_t idbuf[16];
            std::swprintf(idbuf, 16, L"W%d", i);
            ui->worker_id[(size_t)i] = CreateWindowExW(
                0, L"STATIC", idbuf, WS_CHILD | WS_VISIBLE | SS_CENTER,
                0, 0, 52, k_row, hwnd, (HMENU)(INT_PTR)(base + 0), nullptr, nullptr);
            // No PBS_SMOOTH: it is incompatible with PBS_MARQUEE.
            ui->worker_pb[(size_t)i] = CreateWindowExW(
                0, PROGRESS_CLASSW, nullptr, WS_CHILD | WS_VISIBLE,
                0, 0, 100, k_pb_h, hwnd, (HMENU)(INT_PTR)(base + 1), nullptr, nullptr);
            SendMessageW(ui->worker_pb[(size_t)i], PBM_SETRANGE32, 0, k_pb_max);
            ui->worker_phase[(size_t)i] = CreateWindowExW(
                0, L"STATIC", L"", WS_CHILD | WS_VISIBLE | SS_LEFTNOWORDWRAP,
                0, 0, 200, k_row, hwnd, (HMENU)(INT_PTR)(base + 2), nullptr, nullptr);
            set_font(ui->worker_id[(size_t)i], ui->font);
            set_font(ui->worker_phase[(size_t)i], ui->font);
        }

        layout_client(ui);
        SetTimer(hwnd, k_timer_id, k_timer_ms, nullptr);
        return 0;
    }
    case WM_SIZE:
        layout_client(ui);
        return 0;
    case WM_TIMER:
        if (wp == k_timer_id && ui) {
            refresh_ui(ui);
            if (ui->finished && ui->finished()) {
                DestroyWindow(hwnd);
            }
        }
        return 0;
    case WM_CLOSE:
        if (ui && ui->stop_flag)
            ui->stop_flag->store(true, std::memory_order_relaxed);
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        if (ui) {
            KillTimer(hwnd, k_timer_id);
            for (HWND pb : ui->worker_pb)
                set_progress_marquee(pb, false);
            if (ui->font) DeleteObject(ui->font);
            if (ui->font_title) DeleteObject(ui->font_title);
        }
        PostQuitMessage(0);
        return 0;
    default:
        break;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

}  // namespace

void xbkz_ui_run(const BkzUiConfig& cfg, SteadyTimePoint start,
                 std::vector<WorkerStatus>& workers, const GlobalBest& best,
                 std::atomic<bool>& stop_flag,
                 const std::function<bool()>& finished) {
    INITCOMMONCONTROLSEX icc{ sizeof(icc), ICC_PROGRESS_CLASS };
    InitCommonControlsEx(&icc);

    UiState ui{};
    ui.cfg = &cfg;
    ui.start = start;
    ui.workers = &workers;
    ui.best = &best;
    ui.stop_flag = &stop_flag;
    ui.finished = finished;

    const wchar_t* cls = L"XbkzDashboard";
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = cls;
    wc.hCursor = LoadCursorW(nullptr, (LPCWSTR)IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClassW(&wc);

    const int win_h = k_margin * 2 + k_header_rows * k_row + 8
                      + cfg.threads * (k_row + 4) + 20;
    const int win_w = 720;

    HWND hwnd = CreateWindowExW(
        0, cls, L"xbkz \u2014 BKZ lattice reducer",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, win_w, win_h,
        nullptr, nullptr, wc.hInstance, &ui);
    if (!hwnd) return;

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    refresh_ui(&ui);

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
        if (stop_flag.load(std::memory_order_relaxed) && finished && finished())
            break;
    }
}

#endif  // _WIN32
