# VectorBench — Columnar Execution Engine

A C++ columnar execution engine that benchmarks SIMD-vectorized operators against traditional tuple-at-a-time processing. By combining dictionary encoding, Apache Arrow-inspired memory layouts, and 16-wide AVX-512 parallelism, the engine achieves **up to a 15.5x throughput increase** over scalar baselines.

## Optimizations

### Scalar vs. SIMD (AVX-512)
- **Scalar** processes rows one by one, causing high branch prediction pressure on non-uniform data.
- **SIMD** uses 512-bit registers to evaluate 16 integers per instruction.
- **Mask registers** (`__mmask16`) store predicate results directly, eliminating all conditional branches and avoiding pipeline stalls.

### Memory Layouts
- **Arrow-Inspired Bitmaps:** Contiguous value arrays + separate packed validity bitmaps. This doubles cache efficiency by reducing scan stride from 8 bytes/row (interleaved) to 4 bytes/row (dense values).
- **Dictionary Encoding:** Maps strings to `int32` codes, turning expensive string comparisons into fast integer checks. This achieved 8x compression on the test dataset.

## Build

```bash
# Standard build (scalar fallback)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# AVX-512 build (requires Intel Skylake-X / AMD Zen4+ or later)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX512=ON
cmake --build build

./build/vectorbench
```

## Converged Benchmark Results (10M rows, 1000 repeats)

```text
1. Int32 Range Filter: Scalar vs AVX-512 SIMD
   Scalar:   60.65 ms,   164.9 Mrows/s
   SIMD:      4.75 ms,  2107.1 Mrows/s   → 12.8x speedup

2. Apache Arrow-Inspired Columnar Layout
   Row-store:  5.33 ms,  1876.0 Mrows/s
   Arrow:      1.97 ms,  5086.7 Mrows/s   → 2.7x speedup

3. Dictionary Encoding: String Equality Filter
   Raw strings:     32.29 ms,   309.7 Mrows/s
   Dict-encoded:     2.08 ms,  4799.6 Mrows/s   → 15.5x speedup

All correctness checks passed.
```

## Hardware Counters (Linux/x86 only)

The engine wraps `perf_event_open` to measure branch misses, cycles, and instructions. The `measure_perf` template automatically falls back to a no-op on unsupported platforms (like macOS) to ensure portability.

## Project Structure

- `src/vectorbench.cpp` – Scalar and AVX-512 filter implementations.
- `src/arrow_array.cpp` – Row-store vs. Arrow-style columnar layouts.
- `src/dict_column.cpp` – Dictionary encoding for strings.
- `src/perf_counter.cpp` – RAII wrapper for hardware counters.
- `bench/main.cpp` – Benchmark harness with sum validation and correctness checks.