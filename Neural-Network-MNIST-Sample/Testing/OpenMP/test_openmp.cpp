// Compile with OpenMP enabled. This is an installation/correctness smoke test.
#include <iostream>
#include <iomanip>
#include <vector>
#include <omp.h>

#ifndef _OPENMP
#error OpenMP is disabled. Compile with /openmp (MSVC) or -fopenmp (GCC).
#endif

int main() {
    constexpr int count = 1000000;
    const long long expected = 1LL * count * (count + 1) / 2;
    std::vector<int> values(count);
    for (int i = 0; i < count; ++i) values[i] = i + 1;

    long long sequential = 0;
    double start = omp_get_wtime();
    for (int i = 0; i < count; ++i) sequential += values[i];
    const double sequential_seconds = omp_get_wtime() - start;

    long long parallel = 0;
    int threads = 0;
    start = omp_get_wtime();
#pragma omp parallel reduction(+:parallel)
    {
#pragma omp single
        threads = omp_get_num_threads();
#pragma omp for schedule(static)
        for (int i = 0; i < count; ++i) parallel += values[i];
    }
    const double parallel_seconds = omp_get_wtime() - start;

    std::cout << "OpenMP macro: " << _OPENMP
              << "\nAvailable processors: " << omp_get_num_procs()
              << "\nActual threads: " << threads
              << "\nSequential sum: " << sequential
              << "\nParallel sum:   " << parallel
              << "\nExpected sum:   " << expected
              << std::fixed << std::setprecision(6)
              << "\nSequential seconds: " << sequential_seconds
              << "\nParallel seconds:   " << parallel_seconds << '\n';
    const bool passed = sequential == expected && parallel == expected;
    std::cout << (passed ? "PASS" : "FAIL")
              << ": sum verification\n"
              << "Smoke-test timings are not a reliable performance benchmark.\n";
    return passed ? 0 : 1;
}
