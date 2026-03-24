#include "vectorbench.h"
#include "arrow_array.h"
#include "dict_column.h"
#include "perf_counter.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>
#include <string>

using namespace vectorbench;

/**
 * @brief Prints a repeated character sequence to visually segment terminal output.
 * @param c The character used for the divider line.
 * @param width The total character width of the divider.
 */
static void divider(char c = '-', int width = 78) {
    // Generate and print a string of the specified character and length.
    std::cout << std::string(width, c) << "\n";
}

/**
 * @brief Formats and outputs the benchmark metrics for scalar versus SIMD operators.
 * @param r The result structure containing throughput and timing data.
 */
static void print_simd_result(const BenchResult& r) {
    // Format the name, execution time, throughput, and match count into a aligned row.
    std::cout << std::left  << std::setw(52) << r.name
              << std::right << std::setw(9)  << std::fixed << std::setprecision(2)
              << r.elapsed_ms << " ms"
              << std::setw(10) << r.throughput_mrows << " Mrows/s"
              << "  matched=" << r.rows_passed << "\n";
}

/**
 * @brief Formats and outputs the benchmark metrics for physical memory layouts.
 * @param r The result structure containing throughput and timing data.
 */
static void print_arrow_result(const ArrowBenchResult& r) {
    // Output the layout name along with timing and throughput metrics.
    std::cout << std::left  << std::setw(52) << r.name
              << std::right << std::setw(9)  << std::fixed << std::setprecision(2)
              << r.elapsed_ms << " ms"
              << std::setw(10) << r.throughput_mrows << " Mrows/s"
              << "  matched=" << r.rows_passed << "\n";
}

/**
 * @brief Formats and outputs the benchmark metrics for dictionary encoding evaluations.
 * @param r The result structure containing throughput and timing data.
 */
static void print_dict_result(const DictBenchResult& r) {
    // Print the results for the string filtering benchmark.
    std::cout << std::left  << std::setw(52) << r.name
              << std::right << std::setw(9)  << std::fixed << std::setprecision(2)
              << r.elapsed_ms << " ms"
              << std::setw(10) << r.throughput_mrows << " Mrows/s"
              << "  matched=" << r.rows_matched << "\n";
}

/**
 * @brief Converts a raw byte count into a human-readable megabyte string.
 * @param bytes The total memory footprint in bytes.
 * @return Formatted string representing the size in MB.
 */
static std::string mb(size_t bytes) {
    // Divide the byte count by 1e6 and format the result as a string.
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(0) << (bytes / 1e6) << " MB";
    return ss.str();
}

/**
 * @brief Main execution harness for the VectorBench columnar engine test suite.
 * @return Integer status code (0 for success).
 */
int main() {
    // Define the dataset size and predicate bounds for the integer benchmarks.
    constexpr size_t  ROWS    = 10'000'000;
    constexpr int32_t LO      = 200'000;
    constexpr int32_t HI      = 800'000;
    constexpr int32_t MIN_VAL = 0;
    constexpr int32_t MAX_VAL = 1'000'000;

    // Define the string categories and the search target for the dictionary benchmark.
    const std::vector<std::string> CATEGORIES = {
        "Electronics", "Clothing", "Books", "Home & Garden",
        "Sports",      "Toys",     "Food",  "Automotive"
    };
    const std::string FILTER_TARGET = "Electronics";

    std::cout << "VectorBench\n";
    divider('=');
    std::cout << "\n";

    std::cout << "1. Int32 Range Filter: Scalar vs AVX-512 SIMD\n";
    std::cout << "   Rows: " << ROWS / 1'000'000 << "M"
              << "  Predicate: col >= " << LO << " AND col <= " << HI << "\n\n";
    
    // Create the test column and initialize it with random integer data.
    std::cout << "   Generating column... ";
    auto col = generate_column(ROWS, MIN_VAL, MAX_VAL);
    std::cout << "done\n\n";

    // Perform a single iteration of each benchmark to warm up the instruction cache and memory.
    bench_scalar(col, LO, HI, 1);
    bench_simd  (col, LO, HI, 1);

    // Run the scalar and SIMD aggregations and compare the results to ensure correctness.
    {
        auto scalar_sel = scalar_range_filter(col, LO, HI);
        int64_t scalar_sum_val = scalar_sum(col, scalar_sel);

        auto simd_mask = simd_range_filter_mask(col, LO, HI);
        int64_t simd_sum_val = simd_masked_sum(col, simd_mask);

        assert(scalar_sum_val == simd_sum_val);
        std::cout << "   Sum validation: scalar sum = SIMD sum ✓\n\n";
    }

    std::cout << "   " << std::left << std::setw(52) << "Method"
              << std::right << std::setw(9) << "Time"
              << std::setw(14) << "Throughput" << "\n";
    std::cout << "   "; divider();

    // Measure and print the average throughput for the scalar and SIMD range filters.
    auto r_scalar = bench_scalar(col, LO, HI, 1000);
    auto r_simd   = bench_simd  (col, LO, HI, 1000);
    std::cout << "   "; print_simd_result(r_scalar);
    std::cout << "   "; print_simd_result(r_simd);

    // Verify the match counts and calculate the execution speedup.
    assert(r_scalar.rows_passed == r_simd.rows_passed);
    double sp1 = r_scalar.elapsed_ms / r_simd.elapsed_ms;
    std::cout << "\n   Correctness: scalar == simd ✓"
              << "  Speedup: " << std::fixed << std::setprecision(1) << sp1 << "x\n\n";

    // Capture branch and cycle metrics for the scalar and SIMD implementations.
    std::cout << "   Hardware counters (Linux/x86 only):\n";
    auto pc_scalar = measure_perf([&]{ bench_scalar(col, LO, HI, 1); });
    auto pc_simd   = measure_perf([&]{ bench_simd  (col, LO, HI, 1); });
    print_perf_sample("Scalar", pc_scalar);
    print_perf_sample("SIMD  ", pc_simd);
    std::cout << "\n";

    std::cout << "2. Apache Arrow-Inspired Columnar Layout\n";
    std::cout << "   Row-store: interleaved {int32 value, uint8 valid} (8 bytes/row)\n"
              << "   Arrow:     contiguous values[] + packed validity bitmap (4 bytes/row)\n\n";
    
    // Build the row-store and columnar data structures for the layout benchmark.
    std::cout << "   Building structures... ";
    auto row_store = make_row_store  (ROWS, MIN_VAL, MAX_VAL);
    auto arrow_arr = make_arrow_array(ROWS, MIN_VAL, MAX_VAL);
    std::cout << "done\n\n";

    // Log the calculated memory usage of each layout.
    std::cout << "   Memory:\n"
              << "     Row-store : " << mb(row_store.size() * sizeof(RowInt32)) << "\n"
              << "     Arrow     : " << mb(arrow_arr.size_bytes()) << "\n\n";

    // Initialize the memory and cache state for the layout comparison.
    bench_row_store  (row_store, LO, HI, 1);
    bench_arrow_array(arrow_arr, LO, HI, 1);

    std::cout << "   " << std::left << std::setw(52) << "Method"
              << std::right << std::setw(9) << "Time"
              << std::setw(14) << "Throughput" << "\n";
    std::cout << "   "; divider();

    // Execute the layout benchmarks over multiple repetitions.
    auto r_row   = bench_row_store  (row_store, LO, HI, 1000);
    auto r_arrow = bench_arrow_array(arrow_arr, LO, HI, 1000);
    std::cout << "   "; print_arrow_result(r_row);
    std::cout << "   "; print_arrow_result(r_arrow);

    // Calculate the performance difference between the row and columnar layouts.
    assert(r_row.rows_passed == r_arrow.rows_passed);
    double sp2 = r_row.elapsed_ms / r_arrow.elapsed_ms;
    std::cout << "\n   Correctness: row-store == arrow ✓"
              << "  Speedup: " << std::fixed << std::setprecision(1) << sp2 << "x\n\n";

    std::cout << "3. Dictionary Encoding: String Equality Filter\n";
    std::cout << "   " << ROWS / 1'000'000 << "M rows, "
              << CATEGORIES.size() << " unique categories (~"
              << std::fixed << std::setprecision(1)
              << 100.0 / CATEGORIES.size() << "% selectivity)\n\n";
    
    // Generate the raw string data and create the dictionary-encoded equivalent.
    std::cout << "   Building string column and encoding... ";
    auto str_col   = generate_string_column(ROWS, CATEGORIES);
    auto dict_col  = dict_encode(str_col);
    int32_t target_code = dict_col.lookup_code(FILTER_TARGET);
    std::cout << "done\n\n";

    // Output compression statistics for the dictionary encoding.
    size_t raw_b  = dict_col.size_bytes_raw();
    size_t dict_b = dict_col.size_bytes_dict();
    std::cout << "   Encoding stats:\n"
              << "     Cardinality : " << dict_col.cardinality() << " unique values\n"
              << "     Raw strings : ~" << mb(raw_b) << "\n"
              << "     Dict encoded: ~" << mb(dict_b) << "  ("
              << std::fixed << std::setprecision(0)
              << static_cast<double>(raw_b) / dict_b << "x compression)\n"
              << "     Target      : \"" << FILTER_TARGET
              << "\"  code=" << target_code << "\n\n";

    // Run warmup passes for the string filtering logic.
    bench_filter_raw (str_col,  FILTER_TARGET, 1);
    bench_filter_dict(dict_col, target_code,   1);

    std::cout << "   " << std::left << std::setw(52) << "Method"
              << std::right << std::setw(9) << "Time"
              << std::setw(14) << "Throughput" << "\n";
    std::cout << "   "; divider();

    // Compare the execution speed of raw string matching versus integer code matching.
    auto r_raw  = bench_filter_raw (str_col,  FILTER_TARGET, 1000);
    auto r_dict = bench_filter_dict(dict_col, target_code,   1000);
    std::cout << "   "; print_dict_result(r_raw);
    std::cout << "   "; print_dict_result(r_dict);

    // Verify consistency and calculate the resulting speedup.
    assert(r_raw.rows_matched == r_dict.rows_matched);
    double sp3 = r_raw.elapsed_ms / r_dict.elapsed_ms;
    std::cout << "\n   Correctness: raw == dict ✓"
              << "  Speedup: " << std::fixed << std::setprecision(1) << sp3 << "x\n\n";

    // Record hardware performance samples for the dictionary encoding test.
    std::cout << "   Hardware counters (Linux/x86 only):\n";
    auto pc_raw  = measure_perf([&]{ bench_filter_raw (str_col,  FILTER_TARGET, 1); });
    auto pc_dict = measure_perf([&]{ bench_filter_dict(dict_col, target_code,   1); });
    print_perf_sample("Raw strings  ", pc_raw);
    print_perf_sample("Dict encoded ", pc_dict);
    std::cout << "\n";

    divider('=');
    std::cout << "All correctness checks passed.\n";
    return 0;
}