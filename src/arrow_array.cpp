#include "arrow_array.h"
#include <random>
#include <chrono>
#include <algorithm>
#include <cassert>

namespace vectorbench {

/**
 * @brief Populates a row-store array with random data.
 * @param rows Number of rows to generate.
 * @param min_val Minimum integer value.
 * @param max_val Maximum integer value.
 * @param seed Random generation seed.
 * @return The generated row store.
 */
RowStore make_row_store(size_t rows, int32_t min_val, int32_t max_val, uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);
    RowStore store(rows);
    
    // Assign a random value and a valid status bit to every struct in the vector.
    for (auto& r : store) { 
        r.value = dist(rng); 
        r.valid = 1; 
    }
    return store;
}

/**
 * @brief Populates an Arrow-style array with random data and zero nulls.
 * @param rows Number of rows to generate.
 * @param min_val Minimum integer value.
 * @param max_val Maximum integer value.
 * @param seed Random generation seed.
 * @return The generated columnar array.
 */
ArrowInt32Array make_arrow_array(size_t rows, int32_t min_val, int32_t max_val,
                                  uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);
    ArrowInt32Array arr;
    arr.length     = rows;
    arr.null_count = 0; 
    arr.values.resize(rows);
    
    // Size the validity buffer and fill it with 1s to mark all entries as valid.
    arr.validity.resize((rows + 63) / 64, ~uint64_t{0});
    // Populate the contiguous values array with generated integers.
    for (auto& v : arr.values) v = dist(rng);
    
    return arr;
}

/**
 * @brief Computes the total number of valid rows, utilizing the fast path if possible.
 * @return The count of valid rows.
 */
size_t ArrowInt32Array::count_valid() const {
    // If null_count is zero, return the pre-calculated logical length.
    if (null_count == 0) return length;
    
    size_t count = 0;
    for (size_t w = 0; w < validity.size(); ++w) {
        uint64_t word = validity[w];
        size_t base = w * 64;
        
        // Use a bitmask to zero out bits in the last word that exceed the array length.
        if (base + 64 > length) {
            size_t rem = length - base;
            word &= (rem < 64) ? ((1ULL << rem) - 1) : ~uint64_t{0};
        }
        // Accumulate the number of set bits in each 64-bit word.
        count += __builtin_popcountll(word);
    }
    return count;
}

/**
 * @brief Filters a row-store array and returns the number of matches.
 * @param store The input row layout data.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return Count of matching, valid rows.
 */
size_t row_store_range_filter(const RowStore& store, int32_t lo, int32_t hi) {
    size_t count = 0;
    
    // Iterate through the vector of structs and compare each value against the bounds.
    for (const auto& row : store) {
        if (row.value >= lo && row.value <= hi) ++count;
    }
    return count;
}

/**
 * @brief Generates a selection bitmask for an Arrow array using zero-copy scanning.
 * @param arr The input columnar array.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return A packed bitmask array of matches.
 */
std::vector<uint64_t> arrow_range_filter(const ArrowInt32Array& arr,
                                          int32_t lo, int32_t hi) {
    size_t n     = arr.length;
    size_t words = (n + 63) / 64;
    std::vector<uint64_t> result(words, 0);

    if (arr.null_count == 0) {
        // Execute a linear scan across the values and set the corresponding bit in the result word for matches.
        for (size_t i = 0; i < n; ++i) {
            if (arr.values[i] >= lo && arr.values[i] <= hi) {
                result[i / 64] |= (1ULL << (i % 64));
            }
        }
    } else {
        // Verify the validity bit for each index before performing the range comparison.
        for (size_t i = 0; i < n; ++i) {
            if (arr.is_valid(i) && arr.values[i] >= lo && arr.values[i] <= hi) {
                result[i / 64] |= (1ULL << (i % 64));
            }
        }
    }
    return result;
}

/**
 * @brief Filters an Arrow array and returns only the count of matches.
 * @param arr The input columnar array.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return Count of matching, valid rows.
 */
size_t arrow_range_count(const ArrowInt32Array& arr, int32_t lo, int32_t hi) {
    size_t count = 0;
    
    // Branch to the appropriate scan logic based on whether null values are present.
    if (arr.null_count == 0) {
        for (size_t i = 0; i < arr.length; ++i) {
            if (arr.values[i] >= lo && arr.values[i] <= hi) ++count;
        }
    } else {
        for (size_t i = 0; i < arr.length; ++i) {
            if (arr.is_valid(i) && arr.values[i] >= lo && arr.values[i] <= hi) ++count;
        }
    }
    return count;
}

/**
 * @brief Helper to count bits, reserved for optional debug validations.
 * @param mask The packed bitmask array.
 * @param n Total logical rows.
 * @return The total number of matched rows.
 */
[[maybe_unused]] static size_t count_bits(const std::vector<uint64_t>& mask, size_t n) {
    size_t count = 0;
    for (size_t w = 0; w < mask.size(); ++w) {
        uint64_t word = mask[w];
        size_t base = w * 64;
        if (base + 64 > n) {
            size_t rem = n - base;
            word &= (rem < 64) ? ((1ULL << rem) - 1) : ~uint64_t{0};
        }
        count += __builtin_popcountll(word);
    }
    return count;
}

/**
 * @brief Benchmarks the row-oriented memory layout.
 * @param store The input row data.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @param repeats Iteration count for smoothing.
 * @return The averaged performance metrics.
 */
ArrowBenchResult bench_row_store(const RowStore& store, int32_t lo, int32_t hi,
                                  int repeats) {
    // Write results to a volatile variable to prevent the compiler from optimizing out the scan.
    volatile size_t sink = 0; 
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t matched = 0;
    
    for (int r = 0; r < repeats; ++r) {
        matched = row_store_range_filter(store, lo, hi);
        sink    = matched;
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)sink;
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
    return {"Row-store (interleaved {value,valid} stride-8)", store.size(), matched, ms,
            static_cast<double>(store.size()) / ms / 1000.0};
}

/**
 * @brief Benchmarks the columnar Arrow memory layout.
 * @param arr The input columnar data.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @param repeats Iteration count for smoothing.
 * @return The averaged performance metrics.
 */
ArrowBenchResult bench_arrow_array(const ArrowInt32Array& arr, int32_t lo, int32_t hi,
                                    int repeats) {
    volatile size_t sink = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t matched = 0;
    
    for (int r = 0; r < repeats; ++r) {
        // Execute the columnar counting logic across multiple repetitions for timing.
        matched = arrow_range_count(arr, lo, hi);
        sink    = matched;
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)sink;
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
    return {"Arrow col. (dense values[] stride-4, no validity check)", arr.length, matched, ms,
            static_cast<double>(arr.length) / ms / 1000.0};
}

} // namespace vectorbench