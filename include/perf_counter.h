#pragma once
#include <cstdint>
#include <string>

namespace vectorbench {

/**
 * @brief Hardware performance metrics for low-level execution analysis.
 */
struct PerfSample {
    uint64_t branch_misses = 0;   ///< Total branch mispredictions during execution.
    uint64_t cycles = 0;          ///< Total CPU cycles consumed.
    uint64_t instructions = 0;    ///< Total retired instructions.
    bool available = false;       ///< True only on supported Linux/x86 configurations.

    /**
     * @brief Computes average Cycles Per Instruction (CPI).
     * @return The CPI ratio, or 0.0 if counters are unavailable.
     */
    double cpi() const {
        return (available && instructions > 0)
            ? static_cast<double>(cycles) / static_cast<double>(instructions)
            : 0.0;
    }
};

/**
 * @brief Prints formatted hardware metrics to the standard output.
 * @param label A prefix label for the log output.
 * @param s The collected performance sample to print.
 */
void print_perf_sample(const std::string& label, const PerfSample& s);

#ifdef __linux__

/**
 * @brief RAII wrapper for the Linux perf_event_open system call.
 */
class PerfCounter {
public:
    /**
     * @brief Specific hardware events supported by the wrapper.
     */
    enum EventType {
        BRANCH_MISSES,
        CYCLES,
        INSTRUCTIONS,
    };

    /**
     * @brief Initializes a counter for a specific hardware event type.
     * @param event The hardware metric to track.
     */
    explicit PerfCounter(EventType event);
    ~PerfCounter();

    PerfCounter(const PerfCounter&) = delete;
    PerfCounter& operator=(const PerfCounter&) = delete;

    /**
     * @brief Activates the hardware counter to begin tracking.
     */
    void start();
    
    /**
     * @brief Halts tracking and reads the final value from the kernel.
     */
    void stop();
    
    uint64_t value() const { return value_; }
    bool available() const { return fd_ >= 0; }

private:
    int fd_ = -1;
    uint64_t value_ = 0;
};

/**
 * @brief Executes a lambda while measuring branch misses, cycles, and instructions.
 * @tparam Fn Type of the callable execution block.
 * @param fn The target function to benchmark.
 * @return Struct containing execution metrics.
 */
template<typename Fn>
PerfSample measure_perf(Fn&& fn) {
    PerfCounter br(PerfCounter::BRANCH_MISSES);
    PerfCounter cy(PerfCounter::CYCLES);
    PerfCounter in(PerfCounter::INSTRUCTIONS);
    if (!br.available()) { fn(); return PerfSample{}; }
    br.start(); cy.start(); in.start();
    fn();
    in.stop(); cy.stop(); br.stop();
    return PerfSample{br.value(), cy.value(), in.value(), true};
}

#else // !__linux__

/**
 * @brief Fallback wrapper ensuring functional correctness on non-Linux platforms (e.g., macOS/ARM).
 * @tparam Fn Type of the callable execution block.
 * @param fn The target function to benchmark.
 * @return An empty placeholder struct.
 */
template<typename Fn>
PerfSample measure_perf(Fn&& fn) {
    fn(); 
    return PerfSample{};
}

#endif // __linux__

} // namespace vectorbench