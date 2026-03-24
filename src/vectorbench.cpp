#include "vectorbench.h"
#include <random>
#include <chrono>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#  include <immintrin.h>
#  define HAS_AVX512 1
#else
#  define HAS_AVX512 0
#endif

namespace vectorbench {

/**
 * @brief Generates a synthetic dataset of uniformly distributed integers.
 * @param rows Total number of rows to generate.
 * @param min_val Minimum possible integer value.
 * @param max_val Maximum possible integer value.
 * @param seed Random seed for deterministic generation.
 * @return The generated integer column.
 */
Int32Col generate_column(size_t rows, int32_t min_val, int32_t max_val, uint64_t seed) {
    // Initialize the random number generator and define the integer distribution range.
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);
    Int32Col col(rows);

    // Fill the vector with generated values.
    for (auto& v : col) v = dist(rng);
    return col;
}

/**
 * @brief Filters a column using a scalar, tuple-at-a-time loop.
 * @param col The input column of 32-bit integers.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return A vector of row indices that satisfy the predicate.
 */
std::vector<int> scalar_range_filter(const Int32Col& col, int32_t lo, int32_t hi) {
    std::vector<int> out;
    out.reserve(col.size() / 4);
    
    // Scan each element and check if it sits between the lower and upper bounds.
    for (size_t i = 0; i < col.size(); ++i) {
        if (col[i] >= lo && col[i] <= hi) {
            out.push_back(static_cast<int>(i));
        }
    }
    return out;
}

/**
 * @brief Computes a scalar sum over a pre-selected subset of row indices.
 * @param col The input column of 32-bit integers.
 * @param selected A vector of indices identifying the rows to sum.
 * @return The 64-bit sum of the selected rows.
 */
int64_t scalar_sum(const Int32Col& col, const std::vector<int>& selected) {
    int64_t s = 0;
    // Iterate through the index list and accumulate the corresponding column values.
    for (int idx : selected) {
        s += col[idx];
    }
    return s;
}

/**
 * @brief Filters a column using 16-wide AVX-512 SIMD intrinsics.
 * @param col The input column of 32-bit integers.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return A packed bitmask array where bit `i` indicates if row `i` matches.
 */
std::vector<uint64_t> simd_range_filter_mask(const Int32Col& col, int32_t lo, int32_t hi) {
    size_t n = col.size();
    size_t words = (n + 63) / 64;
    std::vector<uint64_t> mask(words, 0);

#if HAS_AVX512
    // Load the scalar bounds into all lanes of 512-bit registers.
    __m512i v_lo = _mm512_set1_epi32(lo);
    __m512i v_hi = _mm512_set1_epi32(hi);

    size_t i = 0;
    for (size_t w = 0; w < words; ++w) {
        uint64_t current_word = 0;
        
        // Process 64 elements per outer loop iteration using four 16-wide SIMD loads.
        for (int lane = 0; lane < 4 && i + 16 <= n; ++lane, i += 16) {
            // Load 16 integers and perform parallel comparisons against the bounds registers.
            __m512i v_data = _mm512_loadu_si512(reinterpret_cast<const void*>(&col[i]));
            
            __mmask16 m_lo  = _mm512_cmpge_epi32_mask(v_data, v_lo);
            __mmask16 m_hi  = _mm512_cmple_epi32_mask(v_data, v_hi);
            __mmask16 m_and = m_lo & m_hi;

            // Combine the 16-bit comparison mask into the current 64-bit result word.
            current_word |= (static_cast<uint64_t>(m_and) << (lane * 16));
        }
        mask[w] = current_word;
    }

    // Handle any remaining elements that do not fit into a full 512-bit register.
    for (; i < n; ++i) {
        if (col[i] >= lo && col[i] <= hi) {
            mask[i / 64] |= (1ULL << (i % 64));
        }
    }
#else
    // Fallback scalar loop to generate the bitmask if AVX-512 is unavailable.
    for (size_t i = 0; i < n; ++i) {
        if (col[i] >= lo && col[i] <= hi) {
            mask[i / 64] |= (1ULL << (i % 64));
        }
    }
#endif

    return mask;
}

/**
 * @brief Aggregates rows mapped to a SIMD selection bitmask.
 * @param col The input column of 32-bit integers.
 * @param mask The packed bitmask dictating which rows to include.
 * @return The 64-bit sum of the selected rows.
 */
int64_t simd_masked_sum(const Int32Col& col, const std::vector<uint64_t>& mask) {
    int64_t sum = 0;
    size_t  n   = col.size();

#if HAS_AVX512
    // Initialize two 512-bit accumulators to zero.
    __m512i acc_lo = _mm512_setzero_si512();
    __m512i acc_hi = _mm512_setzero_si512();

    for (size_t w = 0; w < mask.size(); ++w) {
        uint64_t word = mask[w];
        size_t   base = w * 64;

        for (int lane = 0; lane < 4; ++lane) {
            size_t offset = base + lane * 16;
            if (offset >= n) break;

            // Extract the 16-bit mask for the current lane.
            __mmask16 m = static_cast<__mmask16>((word >> (lane * 16)) & 0xFFFF);

            // Apply a bounds-check mask to the final block to avoid out-of-bounds reads.
            if (offset + 16 > n) {
                size_t valid = n - offset;
                m = m & static_cast<__mmask16>((1u << valid) - 1);
            }

            // Perform a masked load where only bits set in the mask are pulled from memory.
            __m512i v = _mm512_maskz_loadu_epi32(m, &col[offset]);

            // Split the 512-bit vector into two 256-bit halves.
            __m256i v_lo256 = _mm512_castsi512_si256(v);
            __m256i v_hi256 = _mm512_extracti64x4_epi64(v, 1);
            
            // Convert 32-bit integers to 64-bit and add them to the accumulators.
            acc_lo = _mm512_add_epi64(acc_lo, _mm512_cvtepi32_epi64(v_lo256));
            acc_hi = _mm512_add_epi64(acc_hi, _mm512_cvtepi32_epi64(v_hi256));
        }
    }

    // Perform a horizontal addition across the lanes of both accumulators.
    sum = _mm512_reduce_add_epi64(_mm512_add_epi64(acc_lo, acc_hi));
#else
    // Fallback loop using bit manipulation to find and sum only the active indices.
    for (size_t w = 0; w < mask.size(); ++w) {
        uint64_t word = mask[w];
        size_t   base = w * 64;
        
        while (word) {
            int    bit = __builtin_ctzll(word);
            size_t i   = base + bit;
            if (i < n) sum += col[i];
            word &= word - 1; // Clear the trailing set bit.
        }
    }
#endif

    return sum;
}

/**
 * @brief Counts set bits in a validity mask, clamped to the logical row bounds.
 * @param mask The packed bitmask array.
 * @param num_rows Total logical rows, used to mask out tail-end garbage bits.
 * @return The total number of matched rows.
 */
size_t popcount_mask(const std::vector<uint64_t>& mask, size_t num_rows) {
    size_t count = 0;
    for (size_t w = 0; w < mask.size(); ++w) {
        uint64_t word = mask[w];
        size_t base = w * 64;
        
        // Zero out bits in the final 64-bit word that exceed the total row count.
        if (base + 64 > num_rows) {
            size_t valid = num_rows - base;
            word &= (valid < 64) ? ((1ULL << valid) - 1) : ~0ULL;
        }
        // Use the hardware popcount instruction to tally the set bits.
        count += __builtin_popcountll(word);
    }
    return count;
}

/**
 * @brief Benchmarks the scalar filter and summation sequence.
 * @param col The input column data.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @param repeats Number of iterations for timing stability.
 * @return The averaged performance metrics.
 */
BenchResult bench_scalar(const Int32Col& col, int32_t lo, int32_t hi, int repeats) {
    // Store result in a volatile variable to prevent the compiler from optimizing away the loop.
    volatile int64_t sink = 0; 
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<int> sel;
    for (int r = 0; r < repeats; ++r) {
        // Execute the scalar filtering and summation logic.
        sel  = scalar_range_filter(col, lo, hi);
        sink = scalar_sum(col, sel);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    // Calculate the average execution time in milliseconds.
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;

    (void)sink;
    return BenchResult{
        "Scalar (tuple-at-a-time)",
        col.size(), sel.size(), ms,
        static_cast<double>(col.size()) / ms / 1000.0
    };
}

/**
 * @brief Benchmarks the AVX-512 filter and summation sequence.
 * @param col The input column data.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @param repeats Number of iterations for timing stability.
 * @return The averaged performance metrics.
 */
BenchResult bench_simd(const Int32Col& col, int32_t lo, int32_t hi, int repeats) {
    // Prevent compiler optimizations from removing the SIMD execution block.
    volatile int64_t sink = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<uint64_t> mask;
    for (int r = 0; r < repeats; ++r) {
        // Execute the vectorized filtering and summation operations.
        mask    = simd_range_filter_mask(col, lo, hi);
        sink    = simd_masked_sum(col, mask);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    size_t matched = popcount_mask(mask, col.size());
    
    // Compute the average duration across all repetitions.
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;

    std::string name = "SIMD (AVX-512, 16-wide integer parallelism)";
#if !HAS_AVX512
    name = "SIMD (scalar fallback, compile with -mavx512f to enable)";
#endif

    (void)sink;
    return BenchResult{name, col.size(), matched, ms,
                       static_cast<double>(col.size()) / ms / 1000.0};
}

} // namespace vectorbench