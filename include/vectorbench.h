#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace vectorbench {

using Int32Col = std::vector<int32_t>;

/**
 * @brief Standardized metrics for execution engine benchmarks.
 */
struct BenchResult {
    std::string name;             ///< Display name of the benchmarked routine.
    size_t rows_processed;        ///< Total number of rows evaluated.
    size_t rows_passed;           ///< Number of rows satisfying the predicate.
    double elapsed_ms;            ///< Total execution time in milliseconds.
    double throughput_mrows;      ///< Throughput in millions of rows per second.
};

/**
 * @brief Filters a column using a scalar, tuple-at-a-time loop.
 * @param col The input column of 32-bit integers.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return A vector of row indices that satisfy the predicate.
 */
std::vector<int> scalar_range_filter(const Int32Col& col, int32_t lo, int32_t hi);

/**
 * @brief Computes a scalar sum over a pre-selected subset of row indices.
 * @param col The input column of 32-bit integers.
 * @param selected A vector of indices identifying the rows to sum.
 * @return The 64-bit sum of the selected rows.
 */
int64_t scalar_sum(const Int32Col& col, const std::vector<int>& selected);

/**
 * @brief Filters a column using 16-wide AVX-512 SIMD intrinsics.
 * @param col The input column of 32-bit integers.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return A packed bitmask array where bit `i` indicates if row `i` matches.
 */
std::vector<uint64_t> simd_range_filter_mask(const Int32Col& col, int32_t lo, int32_t hi);

/**
 * @brief Aggregates rows mapped to a SIMD selection bitmask.
 * @param col The input column of 32-bit integers.
 * @param mask The packed bitmask dictating which rows to include.
 * @return The 64-bit sum of the selected rows.
 */
int64_t simd_masked_sum(const Int32Col& col, const std::vector<uint64_t>& mask);

/**
 * @brief Counts set bits in a validity mask, clamped to the logical row bounds.
 * @param mask The packed bitmask array.
 * @param num_rows Total logical rows, used to mask out tail-end garbage bits.
 * @return The total number of matched rows.
 */
size_t popcount_mask(const std::vector<uint64_t>& mask, size_t num_rows);

/**
 * @brief Benchmarks the scalar filter and summation sequence.
 * @param col The input column data.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @param repeats Number of iterations for timing stability.
 * @return The averaged performance metrics.
 */
BenchResult bench_scalar(const Int32Col& col, int32_t lo, int32_t hi, int repeats = 5);

/**
 * @brief Benchmarks the AVX-512 filter and summation sequence.
 * @param col The input column data.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @param repeats Number of iterations for timing stability.
 * @return The averaged performance metrics.
 */
BenchResult bench_simd(const Int32Col& col, int32_t lo, int32_t hi, int repeats = 5);

/**
 * @brief Generates a synthetic dataset of uniformly distributed integers.
 * @param rows Total number of rows to generate.
 * @param min_val Minimum possible integer value.
 * @param max_val Maximum possible integer value.
 * @param seed Random seed for deterministic generation.
 * @return The generated integer column.
 */
Int32Col generate_column(size_t rows, int32_t min_val, int32_t max_val, uint64_t seed = 42);

} // namespace vectorbench