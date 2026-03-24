#include "dict_column.h"
#include <random>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#include <numeric>

namespace vectorbench {

/**
 * @brief Resolves a literal string into its corresponding integer dictionary code.
 * @param value The target string.
 * @return The dictionary index, or -1 if the string is not present.
 */
int32_t DictColumn::lookup_code(const std::string& value) const {
    // Perform a linear search through the dictionary vector to find a matching string.
    for (size_t i = 0; i < dictionary.size(); ++i) {
        if (dictionary[i] == value) return static_cast<int32_t>(i);
    }
    return -1;
}

/**
 * @brief Estimates the memory footprint if stored as raw std::string elements.
 * @return Memory footprint in bytes.
 */
size_t DictColumn::size_bytes_raw() const {
    size_t total = 0;
    
    // Iterate through the code array and resolve each back to its original dictionary string.
    for (int32_t code : codes) {
        const std::string& s = dictionary[static_cast<size_t>(code)];
        // Add the base struct size and account for heap allocations for strings exceeding SSO limits.
        total += sizeof(std::string) + (s.size() > 15 ? s.size() + 1 : 0);
    }
    return total;
}

/**
 * @brief Calculates the actual memory footprint of the dictionary and code array.
 * @return Memory footprint in bytes.
 */
size_t DictColumn::size_bytes_dict() const {
    size_t dict_bytes = 0;
    // Calculate the physical memory used by the unique strings in the dictionary.
    for (const auto& s : dictionary) {
        dict_bytes += sizeof(std::string) + (s.size() > 15 ? s.size() + 1 : 0);
    }
    // Sum the dictionary size with the memory consumed by the integer code vector.
    return dict_bytes + codes.size() * sizeof(int32_t);
}

/**
 * @brief Compresses a raw string column into a dictionary-encoded format.
 * @param col The raw string vector.
 * @return The dictionary-encoded column.
 */
DictColumn dict_encode(const StringColumn& col) {
    DictColumn result;
    std::unordered_map<std::string, int32_t> index;
    
    // Map each unique string to a unique integer code based on its appearance order.
    for (const auto& value : col) {
        if (index.find(value) == index.end()) {
            int32_t code = static_cast<int32_t>(result.dictionary.size());
            result.dictionary.push_back(value);
            index[value] = code;
        }
    }
    
    // Replace each original string with its corresponding dictionary code.
    result.codes.reserve(col.size());
    for (const auto& value : col) {
        result.codes.push_back(index[value]);
    }
    return result;
}

/**
 * @brief Generates a synthetic column of strings selected from provided categories.
 * @param rows Total number of strings to generate.
 * @param categories The pool of unique strings to draw from.
 * @param seed Random generation seed.
 * @return The generated string column.
 */
StringColumn generate_string_column(size_t rows,
                                     const std::vector<std::string>& categories,
                                     uint64_t seed) {
    // Setup the random engine and index distribution based on the category count.
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, categories.size() - 1);
    StringColumn col(rows);
    
    // Fill the column with random selections from the provided category list.
    for (auto& v : col) {
        v = categories[dist(rng)];
    }
    return col;
}

/**
 * @brief Evaluates an equality predicate using standard byte-wise string comparisons.
 * @param col The raw string column.
 * @param target The string to match.
 * @return The total number of matches.
 */
size_t string_filter_raw(const StringColumn& col, const std::string& target) {
    size_t count = 0;
    
    // Compare each string in the vector against the target literal.
    for (const auto& s : col) {
        if (s == target) ++count;
    }
    return count;
}

/**
 * @brief Evaluates an equality predicate using fast integer code comparisons.
 * @param col The dictionary-encoded column.
 * @param code The integer code representing the target string.
 * @return The total number of matches.
 */
size_t string_filter_dict(const DictColumn& col, int32_t code) {
    size_t count = 0;
    
    // Scan the code array and increment the counter for every matching integer code.
    for (int32_t c : col.codes) {
        if (c == code) ++count;
    }
    return count;
}

/**
 * @brief Benchmarks the raw string evaluation method.
 * @param col The raw string column.
 * @param target The string to match.
 * @param repeats Iteration count for smoothing.
 * @return The averaged performance metrics.
 */
DictBenchResult bench_filter_raw(const StringColumn& col, const std::string& target,
                                  int repeats) {
    // Assign the result to a volatile variable to prevent loop elimination by the compiler.
    volatile size_t sink = 0; 
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t matched = 0;
    
    for (int r = 0; r < repeats; ++r) {
        // Run the byte-wise string equality filter and capture the match count.
        matched = string_filter_raw(col, target);
        sink    = matched;
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)sink;
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
    return {"Raw strings (std::string == per row)", col.size(), matched, ms,
            static_cast<double>(col.size()) / ms / 1000.0};
}

/**
 * @brief Benchmarks the dictionary-encoded evaluation method.
 * @param col The dictionary-encoded column.
 * @param code The target integer code.
 * @param repeats Iteration count for smoothing.
 * @return The averaged performance metrics.
 */
DictBenchResult bench_filter_dict(const DictColumn& col, int32_t code,
                                   int repeats) {
    // Force the execution of the dictionary filter using a volatile sink.
    volatile size_t sink = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t matched = 0;
    
    for (int r = 0; r < repeats; ++r) {
        // Execute the integer-based equality check across the encoded codes.
        matched = string_filter_dict(col, code);
        sink    = matched;
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)sink;
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
    return {"Dict encoded (int32 code == per row)", col.num_rows(), matched, ms,
            static_cast<double>(col.num_rows()) / ms / 1000.0};
}

} // namespace vectorbench