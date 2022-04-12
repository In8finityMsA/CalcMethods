#include "benchmark_matrix.h"

BenchResult BenchmarkInverse(size_t times_repeat, size_t size) {
    BenchResult perf{};

    for (size_t i = 0; i < times_repeat; i++) {
        auto m = generateRealMatrix(size, N, i);

        auto start = std::chrono::high_resolution_clock::now();
        auto inverse = MatrixUtils::Inverse(m);
        auto end = std::chrono::high_resolution_clock::now();
        perf.time1 += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        auto identity = Matrix<double>::GetIdentity(size);
        auto result = m * inverse;
        auto delta = (result - identity).CubicNorm();
        if (delta > 0.001) {
            std::cout << "Delta: " << delta << std::endl;
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

RelaxBenchResult BenchmarkRelax(double relax_param, size_t times_repeat, size_t size) {
    RelaxBenchResult perf{};

    for (size_t i = 0; i < times_repeat; i++) {
        auto m = generateRealMatrix(size, N, i);
        auto vec_x = generateRealVector(size, N, i);
        auto vec_b = m * vec_x;

        auto start = std::chrono::high_resolution_clock::now();
        auto relax_res = MatrixUtils::Relaxation(m, vec_b, 1.0, epsilon);
        auto end = std::chrono::high_resolution_clock::now();
        perf.br.time1 += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        double delta = (relax_res.answer - vec_x).OctahedronNorm();
        if (delta > 0.001) {
            std::cout << "Delta: " << delta << std::endl;
        }
        perf.br.avg_delta += delta;
        perf.br.min_delta = std::min(delta, perf.br.min_delta);
        perf.br.max_delta = std::max(delta, perf.br.max_delta);

        size_t iter_count = relax_res.deltas.size();
        perf.avg_iters += iter_count;
        perf.min_iters = std::min(iter_count, perf.min_iters);
        perf.max_iters = std::max(iter_count, perf.max_iters);
    }
    perf.br.time1 /= times_repeat;
    perf.br.time2 /= times_repeat;
    perf.br.avg_delta /= times_repeat;
    perf.avg_iters /= times_repeat;

    return perf;
}

BenchResult BenchmarkGauss(size_t times_repeat, size_t size) {
    BenchResult perf;

    for (size_t i = 0; i < times_repeat; i++) {
        auto m = generateRealMatrix(size, N, i);
        auto vec_x = generateRealVector(size, N, i);
        auto vec_b = m * vec_x;

        auto start = std::chrono::high_resolution_clock::now();
        auto res = MatrixUtils::LinearSolve(m, vec_b);
        auto end = std::chrono::high_resolution_clock::now();
        perf.time1 += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        double delta = (res - vec_x).OctahedronNorm();
        if (delta > 0.001) {
            std::cout << "Delta: " << delta << std::endl;
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