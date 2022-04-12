#pragma once
#include "matrix.h"
#include "matrix_generator.h"
#include "MatrixUtils.h"
#include "vector.h"
#include <chrono>
#include <iostream>
#include <float.h>
#include "constants.h"

struct BenchResult {
    long long time1 = 0;
    long long time2 = 0;
    double min_delta = DBL_MAX;
    double avg_delta = 0;
    double max_delta = 0;
};

struct RelaxBenchResult {
    BenchResult br;
    size_t min_iters = SIZE_MAX;
    double avg_iters = 0;
    size_t max_iters = 0;
};

template<typename Dec, typename Solv>
BenchResult BenchmarkDecomposition(Dec decompose, Solv lin_solve, size_t times_repeat, size_t size) {
    BenchResult perf;

    for (size_t i = 0; i < times_repeat; i++) {
        auto m = generateRealMatrix(size, N, i);
        auto vec_x = generateRealVector(size, N, i);
        auto vec_b = m * vec_x;

        auto start = std::chrono::high_resolution_clock::now();
        auto dense = decompose(m);
        auto end = std::chrono::high_resolution_clock::now();
        perf.time1 += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        auto start2 = std::chrono::high_resolution_clock::now();
        auto res = lin_solve(dense, vec_b);
        auto end2 = std::chrono::high_resolution_clock::now();
        perf.time2 += std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

        double delta = (res - vec_x).CubicNorm();
        if (delta > 0.001) {
            std::cout << "Delta warning\n";
        }
        perf.avg_delta += delta;
        perf.min_delta = std::min(delta, perf.min_delta);
        perf.max_delta = std::max(delta, perf.max_delta);
    }
    perf.time1 /= times_repeat;
    perf.time2 /= times_repeat;
    perf.avg_delta /= times_repeat;

    return perf;
}

BenchResult BenchmarkInverse(size_t times_repeat, size_t size);

RelaxBenchResult BenchmarkRelax(double relax_param, size_t times_repeat, size_t size);

BenchResult BenchmarkGauss(size_t times_repeat, size_t size);
