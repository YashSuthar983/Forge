#pragma once
// The process's memory as the kernel accounts it (/proc/self/status):
// resident now, resident peak, and the virtual size an address-space limit
// (ulimit -v, RLIMIT_AS) is checked against.  For --verbose stage lines,
// which is how a run's memory is followed stage by stage (a2864 ended at a
// 5.4 GB peak with no trace of which stage held what).  Zeros where /proc
// is unavailable.
#include <cstdio>

namespace forge::core {

struct ProcessMemory {
    double rss_gb = 0.0;
    double peak_gb = 0.0;
    double vm_gb = 0.0;
};

inline ProcessMemory process_memory() {
    ProcessMemory pm;
    if (std::FILE* fp = std::fopen("/proc/self/status", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, fp)) {
            unsigned long long kb = 0;
            if (std::sscanf(line, "VmRSS: %llu kB", &kb) == 1) pm.rss_gb = static_cast<double>(kb) / 1048576.0;
            else if (std::sscanf(line, "VmHWM: %llu kB", &kb) == 1) pm.peak_gb = static_cast<double>(kb) / 1048576.0;
            else if (std::sscanf(line, "VmSize: %llu kB", &kb) == 1) pm.vm_gb = static_cast<double>(kb) / 1048576.0;
        }
        std::fclose(fp);
    }
    return pm;
}

}  // namespace forge::core
