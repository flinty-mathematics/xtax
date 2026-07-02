// stop_signal.hpp: one shared copy of the interrupt / stop plumbing used by the
// congruence annealers (xtax, xdual) and the BKZ reducer (xbkz).
//
// Two flags with distinct roles, adopting the xbkz semantics:
//   g_stop        - the flag hot loops and worker pools poll to end early. It is
//                   set by Ctrl-C and by the wall-clock monitor.
//   g_interrupted - set only by Ctrl-C, so main can tell an interrupted run from
//                   one that simply reached its time budget, and print / write
//                   the best result accordingly.
//
// The first Ctrl-C sets both flags and prints one acknowledgement using a write
// that is safe from a signal or console-control handler. A second Ctrl-C is
// swallowed (the handler stays installed and returns TRUE on Windows), so it can
// never kill the process mid-write and truncate an output CSV. Shutdown is
// always graceful: control returns to main, which joins the workers and writes
// the best result found so far.
//
// Header-only: the globals are inline variables and the functions are inline, so
// each executable's single translation unit gets exactly one definition.

#pragma once

#include <atomic>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <unistd.h>
#endif

// Set on Ctrl-C only, so main can report that the run was interrupted rather
// than finishing its time budget.
inline std::atomic<bool> g_interrupted{ false };

// The single stop flag the hot loops and worker pools poll (set by Ctrl-C and by
// the time-budget monitor). Lock-free for the handler.
inline std::atomic<bool> g_stop{ false };

// Printed once on the first interrupt to acknowledge it immediately.
inline const char k_interrupt_msg[] =
    "\n[interrupt received, finishing and writing the best result so far...]\n";

// Record an interrupt and acknowledge it. Only the first interrupt prints, using
// a write that is safe to call from a signal or console-control handler.
inline void signal_stop() {
    if (!g_stop.exchange(true, std::memory_order_relaxed)) {
        g_interrupted.store(true, std::memory_order_relaxed);
#if defined(_WIN32)
        DWORD wrote = 0;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), k_interrupt_msg,
                  (DWORD)(sizeof(k_interrupt_msg) - 1), &wrote, nullptr);
#else
        ssize_t r = write(STDERR_FILENO, k_interrupt_msg, sizeof(k_interrupt_msg) - 1);
        (void)r;
#endif
    }
}

#if defined(_WIN32)
// On Windows std::signal(SIGINT) resets to the default terminator before the
// handler runs, so a second Ctrl-C would kill the process before the workers are
// joined and the best result is written. SetConsoleCtrlHandler stays installed
// and, by returning TRUE, suppresses the default terminator so shutdown is
// always graceful. The handler runs on its own thread, so it just flips the flags.
inline BOOL WINAPI console_ctrl_handler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) {
        signal_stop();
        return TRUE;
    }
    return FALSE;
}
#else
inline void handle_interrupt(int) { signal_stop(); }
#endif

inline void install_signal_handlers() {
#if defined(_WIN32)
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    std::signal(SIGINT, handle_interrupt);
#endif
}
