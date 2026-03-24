#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace vectorbench {

/**
 * @brief Baseline row-oriented physical layout.
 */
struct RowInt32 {
    int32_t value;   ///< The actual integer data payload.
    uint8_t valid;   ///< 1 indicates a live row, 0 indicates a null.
};

using RowStore = std::vector<RowInt32>;

/**
 * @brief Columnar physical layout inspired by Apache Arrow.
 */
struct ArrowInt32Array {
    std::vector<int32_t> values;     ///< Contiguous array of raw integer values.
    std::vector<uint64_t> validity;  ///< Packed bitmask indicating null status.
    size_t length;                   ///< Total number of logical rows.
    size_t null_count;               ///< A count of 0 allows for bypassed validity checks (fast path).

    /**
     * @brief Extracts the validity status of a specific row.
     * @param i The row index.
     * @return True if the row is valid, false if null.
     */
    bool is_valid(size_t i) const {
        return (validity[i / 64] >> (i % 64)) & 1ULL;
    }

    /**
     * @brief Computes the total number of valid rows, utilizing the fast path if possible.
     * @return The count of valid rows.
     */
    size_t count_valid() const;
    
    /**
     * @brief Calculates the total heap allocation of the array buffers.
     * @return Memory footprint in bytes.
     */
    size_t size_bytes() const {
        return values.size() * sizeof(int32_t) + validity.size() * sizeof(uint64_t);
    }
};

/**
 * @brief Performance result for layout comparisons.
 */
struct ArrowBenchResult {
    std::string name;             ///< Descriptive name of the layout benchmark.
    size_t rows_processed;        ///< Total number of rows evaluated.
    size_t rows_passed;           ///< Number of rows matching the filter.
    double elapsed_ms;            ///< Execution time in milliseconds.
    double throughput_mrows;      ///< Millions of rows processed per second.
};

/**
 * @brief Populates a row-store array with random data.
 * @param rows Number of rows to generate.
 * @param min_val Minimum integer value.
 * @param max_val Maximum integer value.
 * @param seed Random generation seed.
 * @return The generated row store.
 */
RowStore make_row_store(size_t rows, int32_t min_val, int32_t max_val, uint64_t seed = 42);

/**
 * @brief Populates an Arrow-style array with random data and zero nulls.
 * @param rows Number of rows to generate.
 * @param min_val Minimum integer value.
 * @param max_val Maximum integer value.
 * @param seed Random generation seed.
 * @return The generated columnar array.
 */
ArrowInt32Array make_arrow_array(size_t rows, int32_t min_val, int32_t max_val, uint64_t seed = 42);

/**
 * @brief Filters a row-store array and returns the number of matches.
 * @param store The input row layout data.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return Count of matching, valid rows.
 */
size_t row_store_range_filter(const RowStore& store, int32_t lo, int32_t hi);

/**
 * @brief Generates a selection bitmask for an Arrow array using zero-copy scanning.
 * @param arr The input columnar array.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return A packed bitmask array of matches.
 */
std::vector<uint64_t> arrow_range_filter(const ArrowInt32Array& arr, int32_t lo, int32_t hi);

/**
 * @brief Filters an Arrow array and returns only the count of matches.
 * @param arr The input columnar array.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return Count of matching, valid rows.
 */
size_t arrow_range_count(const ArrowInt32Array& arr, int32_t lo, int32_t hi);

/**
 * @brief Benchmarks the row-oriented memory layout.
 * @param store The input row data.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @param repeats Iteration count for smoothing.
 * @return The averaged performance metrics.
 */
ArrowBenchResult bench_row_store(const RowStore& store, int32_t lo, int32_t hi, int repeats = 5);

/**
 * @brief Benchmarks the columnar Arrow memory layout.
 * @param arr The input columnar data.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @param repeats Iteration count for smoothing.
 * @return The averaged performance metrics.
 */
ArrowBenchResult bench_arrow_array(const ArrowInt32Array& arr, int32_t lo, int32_t hi, int repeats = 5);

} // namespace vectorbench