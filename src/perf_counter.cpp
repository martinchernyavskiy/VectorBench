#include "perf_counter.h"
#include <iostream>
#include <iomanip>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <cstdio>

namespace vectorbench {

/**
 * @brief Thin internal wrapper around the Linux perf_event_open syscall.
 * @param attr Pointer to the hardware event configuration structure.
 * @param pid Process ID to monitor (0 indicates the current process).
 * @param cpu CPU core to monitor (-1 indicates any CPU).
 * @param group_fd File descriptor of the group leader (-1 for independent counters).
 * @param flags Additional configuration flags for the syscall.
 * @return The file descriptor for the hardware counter, or -1 on failure.
 */
static int perf_event_open(struct perf_event_attr* attr, pid_t pid, int cpu,
                            int group_fd, unsigned long flags) {
    // Perform a raw system call to open a performance monitoring file descriptor.
    return static_cast<int>(syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags));
}

/**
 * @brief Initializes a counter for a specific hardware event type.
 * @param event The hardware metric to track.
 */
PerfCounter::PerfCounter(EventType event) {
    struct perf_event_attr pe{};
    pe.type           = PERF_TYPE_HARDWARE;
    pe.size           = sizeof(pe);
    pe.disabled       = 1;
    // Set attributes to exclude kernel-space and hypervisor-level events from the tally.
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;

    // Map the requested event type to the corresponding Linux kernel constant.
    switch (event) {
        case BRANCH_MISSES:  pe.config = PERF_COUNT_HW_BRANCH_MISSES;  break;
        case CYCLES:         pe.config = PERF_COUNT_HW_CPU_CYCLES;     break;
        case INSTRUCTIONS:   pe.config = PERF_COUNT_HW_INSTRUCTIONS;   break;
    }

    // Attempt to open the counter for the current process across all CPUs.
    fd_ = perf_event_open(&pe, 0, -1, -1, 0);
}

PerfCounter::~PerfCounter() {
    // Close the file descriptor to release the associated kernel resources.
    if (fd_ >= 0) ::close(fd_);
}

/**
 * @brief Activates the hardware counter to begin tracking.
 */
void PerfCounter::start() {
    if (fd_ < 0) return;
    // Zero out the existing counter value and enable hardware event tracking.
    ioctl(fd_, PERF_EVENT_IOC_RESET,  0);
    ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
}

/**
 * @brief Halts tracking and reads the final value from the kernel.
 */
void PerfCounter::stop() {
    if (fd_ < 0) return;
    // Stop the event monitor and read the resulting 64-bit count into the value buffer.
    ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
    if (::read(fd_, &value_, sizeof(value_)) != sizeof(value_)) {
        value_ = 0;
    }
}

} // namespace vectorbench

#endif // __linux__

namespace vectorbench {

/**
 * @brief Prints formatted hardware metrics to the standard output.
 * @param label A prefix label for the log output.
 * @param s The collected performance sample to print.
 */
void print_perf_sample(const std::string& label, const PerfSample& s) {
    // Validate if performance counters are available for the current architecture before outputting.
    if (!s.available) {
        std::cout << "  " << label
                  << ": perf counters not available on this platform "
                     "(Linux/x86 required)\n";
        return;
    }

    // Output formatted branch misses, cycles, instructions, and the calculated CPI ratio.
    std::cout << "  " << label << "\n"
              << "    branch-misses : " << s.branch_misses << "\n"
              << "    cycles        : " << s.cycles << "\n"
              << "    instructions  : " << s.instructions << "\n"
              << "    CPI           : " << std::fixed << std::setprecision(3)
              << s.cpi() << "\n";
}

} // namespace vectorbench