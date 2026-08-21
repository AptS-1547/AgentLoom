// Best-effort in-process minidump writer for Windows tools.
// Install crash_dump::Install() at the top of main() and a .dmp file will be
// written to the current working directory on fatal failure.
//
// Coverage:
//   - SEH (access violation, divide-by-zero, etc.)
//   - Vectored exception handler with priority=1 — fires before SEH and is
//     not always bypassed by __fastfail (STATUS_STACK_BUFFER_OVERRUN 0xC0000409)
//   - C signals (SIGABRT, SIGSEGV, SIGILL, SIGFPE)
//   - CRT invalid parameter handler
//   - std::terminate
//
// Caveat: stack buffer overrun corrupts the stack, so the in-process writer
// may itself fail. If no .dmp appears, enable WER LocalDumps as a backstop:
//   reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /v DumpFolder /t REG_EXPAND_SZ /d "%TEMP%\dumps" /f
#pragma once

#include <string>

#ifdef _WIN32

#include <windows.h>
#include <dbghelp.h>
#include <crtdbg.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <iostream>

#pragma comment(lib, "dbghelp.lib")

namespace crash_dump {


inline std::atomic<int>& WriterGate() {
    static std::atomic<int> gate{0};
    return gate;
}

inline std::string& DumpDir() {
    static std::string dir;
    return dir;
}

inline std::string MakeDumpPath(DWORD code) {
    char exe[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    std::string base(exe);
    auto slash = base.find_last_of("\\/");
    std::string name = (slash != std::string::npos) ? base.substr(slash + 1) : base;
    auto dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);

    auto t = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &t);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm);

    char codebuf[16];
    std::snprintf(codebuf, sizeof(codebuf), "%08lX", code);

    std::string dir = DumpDir().empty() ? std::string(".") : DumpDir();
    return dir + "\\crash_" + name + "_" + ts + "_pid" +
           std::to_string(GetCurrentProcessId()) + "_" + codebuf + ".dmp";
}

inline void WriteDump(EXCEPTION_POINTERS* ep, DWORD code, const char* origin) {
    // Re-entrant guard: if the dumper itself faults, do not recurse.
    if (WriterGate().fetch_add(1) != 0) {
        return;
    }

    auto path = MakeDumpPath(code);
    HANDLE h = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "[crash_dump] CreateFile failed for %s err=%lu\n",
                     path.c_str(), GetLastError());
        std::fflush(stderr);
        return;
    }

    MINIDUMP_EXCEPTION_INFORMATION mei{};
    mei.ThreadId = GetCurrentThreadId();
    mei.ExceptionPointers = ep;
    mei.ClientPointers = FALSE;

    auto type = static_cast<MINIDUMP_TYPE>(
        MiniDumpWithDataSegs |
        MiniDumpWithThreadInfo |
        MiniDumpWithUnloadedModules |
        MiniDumpWithIndirectlyReferencedMemory |
        MiniDumpWithProcessThreadData);

    BOOL ok = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(),
                                h, type, ep ? &mei : nullptr, nullptr, nullptr);
    DWORD err = ok ? 0 : GetLastError();
    CloseHandle(h);

    std::fprintf(stderr, "[crash_dump] %s code=0x%08lX origin=%s -> %s%s\n",
                 ok ? "WROTE" : "FAILED",
                 code, origin, path.c_str(),
                 ok ? "" : (" err=" + std::to_string(err)).c_str());
    std::fflush(stderr);
}

inline LONG WINAPI SehFilter(EXCEPTION_POINTERS* ep) {
    WriteDump(ep, ep ? ep->ExceptionRecord->ExceptionCode : 0, "SEH");
    return EXCEPTION_EXECUTE_HANDLER;
}

inline LONG CALLBACK VectoredFilter(EXCEPTION_POINTERS* ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    // Filter out non-fatal exceptions (C++ throw, debug breakpoints, etc.)
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:
        case EXCEPTION_STACK_OVERFLOW:
        case EXCEPTION_ILLEGAL_INSTRUCTION:
        case EXCEPTION_INT_DIVIDE_BY_ZERO:
        case EXCEPTION_PRIV_INSTRUCTION:
        case EXCEPTION_IN_PAGE_ERROR:
        case 0xC0000409:  // STATUS_STACK_BUFFER_OVERRUN / __fastfail
        case 0xC0000374:  // STATUS_HEAP_CORRUPTION
            WriteDump(ep, code, "vectored");
            break;
        default:
            break;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

inline void OnSignal(int sig) {
    WriteDump(nullptr, static_cast<DWORD>(sig), "signal");
    _exit(3);
}

inline void OnInvalidParam(const wchar_t*, const wchar_t*, const wchar_t*,
                           unsigned, uintptr_t) {
    WriteDump(nullptr, 0xC0000409, "invalid_param");
    _exit(3);
}

inline void OnTerminate() {
    WriteDump(nullptr, 0, "std::terminate");
    _exit(3);
}

inline void Install(const std::string& dump_dir = ".") {
    DumpDir() = dump_dir;

    // Suppress Windows error dialog so the test fails fast in CI.
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);

    // Vectored handler runs before SEH; priority=1 puts us first in line.
    AddVectoredExceptionHandler(1, VectoredFilter);
    SetUnhandledExceptionFilter(SehFilter);

    std::signal(SIGABRT, OnSignal);
    std::signal(SIGSEGV, OnSignal);
    std::signal(SIGILL, OnSignal);
    std::signal(SIGFPE, OnSignal);

    _set_invalid_parameter_handler(OnInvalidParam);
    _CrtSetReportMode(_CRT_ASSERT, 0);

    std::set_terminate(OnTerminate);

    std::fprintf(stderr, "[crash_dump] installed, dump dir=%s\n", DumpDir().c_str());
    std::fflush(stderr);
}

}  // namespace crash_dump

#else
namespace crash_dump {
inline void Install(const std::string& = ".") {}
}  // namespace crash_dump
#endif
