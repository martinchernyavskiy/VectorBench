#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace vectorbench {

using StringColumn = std::vector<std::string>;

/**
 * @brief Maps strings to integer codes to accelerate low-cardinality string evaluation.
 */
struct DictColumn {
    std::vector<std::string> dictionary;  ///< Array of unique string values.
    std::vector<int32_t> codes;           ///< Array mapping each row to an index in the dictionary.

    // Returns the total number of physical rows.
    size_t num_rows() const { return codes.size(); }
    
    // Returns the number of unique strings in the column.
    size_t cardinality() const { return dictionary.size(); }

    /**
     * @brief Resolves a literal string into its corresponding integer dictionary code.
     * @param value The target string.
     * @return The dictionary index, or -1 if the string is not present.
     */
    int32_t lookup_code(const std::string& value) const;

    /**
     * @brief Estimates the memory footprint if stored as raw std::string elements.
     * @return Memory footprint in bytes.
     */
    size_t size_bytes_raw() const;
    
    /**
     * @brief Calculates the actual memory footprint of the dictionary and code array.
     * @return Memory footprint in bytes.
     */
    size_t size_bytes_dict() const;
};

/**
 * @brief Result metrics for dictionary benchmarks.
 */
struct DictBenchResult {
    std::string name;             ///< Descriptive name of the benchmark variant.
    size_t rows_processed;        ///< Total number of strings evaluated.
    size_t rows_matched;          ///< Total number of strings matching the target.
    double elapsed_ms;            ///< Execution time in milliseconds.
    double throughput_mrows;      ///< Millions of rows processed per second.
};

/**
 * @brief Compresses a raw string column into a dictionary-encoded format.
 * @param col The raw string vector.
 * @return The dictionary-encoded column.
 */
DictColumn dict_encode(const StringColumn& col);

/**
 * @brief Generates a synthetic column of strings selected from provided categories.
 * @param rows Total number of strings to generate.
 * @param categories The pool of unique strings to draw from.
 * @param seed Random generation seed.
 * @return The generated string column.
 */
StringColumn generate_string_column(size_t rows, const std::vector<std::string>& categories, uint64_t seed = 42);

/**
 * @brief Evaluates an equality predicate using standard byte-wise string comparisons.
 * @param col The raw string column.
 * @param target The string to match.
 * @return The total number of matches.
 */
size_t string_filter_raw(const StringColumn& col, const std::string& target);

/**
 * @brief Evaluates an equality predicate using fast integer code comparisons.
 * @param col The dictionary-encoded column.
 * @param code The integer code representing the target string.
 * @return The total number of matches.
 */
size_t string_filter_dict(const DictColumn& col, int32_t code);

/**
 * @brief Benchmarks the raw string evaluation method.
 * @param col The raw string column.
 * @param target The string to match.
 * @param repeats Iteration count for smoothing.
 * @return The averaged performance metrics.
 */
DictBenchResult bench_filter_raw(const StringColumn& col, const std::string& target, int repeats = 5);

/**
 * @brief Benchmarks the dictionary-encoded evaluation method.
 * @param col The dictionary-encoded column.
 * @param code The target integer code.
 * @param repeats Iteration count for smoothing.
 * @return The averaged performance metrics.
 */
DictBenchResult bench_filter_dict(const DictColumn& col, int32_t code, int repeats = 5);

} // namespace vectorbench