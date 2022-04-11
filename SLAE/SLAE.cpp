#include <iostream>
#include "matrix.h"
#include "matrix_generator.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include "MatrixUtils.h"
#include "benchmark_matrix.h"
#include <functional>
#include "constants.h"

#define MUSEC_TO_SEC(microseconds) microseconds / 1000000.0\

using namespace std;

struct CondResult {
    Matrix<double> max_cond_matrix;
    double min_cond = DBL_MAX;
    double avg_cond = 0;
    double max_cond = 0;
};

CondResult GetMatrixConditions(size_t times_repeat, size_t size) {
    CondResult res{};

    for (size_t i = 0; i < times_repeat; i++) {
        auto m = generateRealMatrix(size, N, i);
        
        auto inv = MatrixUtils::Inverse(m);
        auto cond = m.CubicNorm() * inv.CubicNorm();

        res.avg_cond += cond;
        res.min_cond = std::min(cond, res.min_cond);
        if (res.max_cond < cond) {
            res.max_cond = cond;
            res.max_cond_matrix = m;
        }
    }
    res.avg_cond /= times_repeat;
    return res;
}

int main()
{
    const size_t n = 3;
    //auto mat = generateRealMatrix(n, VARIANT);
    //auto mat = Matrix<double>(3);
    std::vector<std::vector<double>> A1_ = { {pow(N, 2) + 15, N - 1, -1, -2},
                                            {N - 1, -15 - pow(N, 2), -N + 4, -4},
                                            {-1, -N + 4, pow(N,2) + 8, -N},
                                            {-2, -4, -N, pow(N,2) + 10} };

    std::vector<std::vector<double>> A2_ = {
        {1,            1 + N,           2 + N,           3 + N,             4 + N,            5 + N,            6 + N,             7 + N},
        {100 * N,      1000 * N,        10000 * N,       100000 * N,        -1000 * N,        -10000 * N,       -100000 * N,       1},
        {N,            -1 + N,          -2 + N,          -3 + N,            -4 + N,           -5 + N,           -6 + N,             -7 + N},
        {N - 1000,     10 * N - 1000,   100 * N - 1000,  1000 * N - 1000,   10000 * N - 1000, -N,               -N + 1,            -N + 2},
        {-2 * N,       0,               -1,              -2,                -3,               -4,               -5,                -6},
        {N - 2019,     -N + 2020,       N - 2021,        -N + 2022,         N - 2023,         -N + 2024,        N - 2025,          -N + 2026},
        {2 * N - 2000, 4 * N - 2005,    8 * N - 2010,    16 * N - 2015,     32 * N - 2020,    2019 * N,         -2020 * N,         2021 * N},
        {1020 - 2 * N, -2924 + 896 * N, 1212 + 9808 * N, -2736 + 98918 * N, 1404 - 11068 * N, -1523 - 8078 * N, 2625 - 102119 * N, -1327 + 1924 * N},
    };
    Matrix<double> A2(A2_);

    //A2.GetTranspose().print();
    std::cout << "\n---------MAT--------" << endl;
    Matrix<double> mat = (A2.GetTranspose() * A2);
    //Matrix<double> mat = generateRealMatrix(3, N);
    mat.print();

    std::cout << "\nVector y:\n";
    auto vec_y = generateRealVector(mat.Size(), N);
    vec_y.print();

    std::cout << "\nVector b:\n";
    auto vec_b = mat * vec_y;
    vec_b.print();


    std::cout << "\n---------GAUSS--------" << endl;
    auto vec_answer = MatrixUtils::LinearSolve(mat, vec_b);
    vec_answer.print();

    std::cout << "\n---------LUP--------" << endl;
    std::cout << "Before LU mat A:\n";
    mat.print();
    auto lup_dense = MatrixUtils::DecompositionLUP(mat);
    auto lup = MatrixUtils::UndenseLUP(lup_dense);
    auto res = lup.P * lup.L * lup.U;
    std::cout << "\n**Mat P:\n";
    lup.P.print();
    std::cout << "\n**Mat L:\n";
    lup.L.print();
    std::cout << "\n**Mat U:\n";
    lup.U.print();
    std::cout << "\n**LU mat A:\n";
    res.print();
    auto vec_answer_lu = MatrixUtils::LinearSolveWithLUP(lup_dense, vec_b);
    vec_answer_lu.print();

    std::cout << "\n---------LDLT--------" << endl;
    std::cout << "**Before LDLT mat A:\n";
    mat.print();
    auto ldlt_dense = MatrixUtils::DecompositionLDLT(mat);
    auto ldlt = MatrixUtils::UndenseLDLT(ldlt_dense);
    auto res_ldlt = ldlt.L * ldlt.D * ldlt.LT;
    std::cout << "\n**Mat L:\n";
    ldlt.L.print();
    std::cout << "\n**Mat D:\n";
    ldlt.D.print();
    std::cout << "\n**Mat LT:\n";
    ldlt.LT.print();
    std::cout << "\n**LDLT mat A:\n";
    res_ldlt.print();
    auto vec_answer_ldlt = MatrixUtils::LinearSolveWithLDLT(ldlt_dense, vec_b);
    vec_answer_ldlt.print();

    std::cout << "\n---------Relax--------" << endl;
    auto vec_relax = MatrixUtils::Relaxation(mat, vec_b, 1.0, 10e-10);
    std::cout << "**Relax answer:\n";
    vec_relax.print();


    std::cout << "\n---------Conditions--------" << endl;
    auto cond_res = GetMatrixConditions(TEST_COUNT, TEST_SIZE);
    cout << "Min cond: " << cond_res.min_cond << ", Avg cond: " << cond_res.avg_cond << ", Max cond: " << cond_res.max_cond << endl;
    //cond_res.max_cond_matrix.print();
    cout << "Max mat cond: " << cond_res.max_cond_matrix.CubicNorm() * MatrixUtils::Inverse(cond_res.max_cond_matrix).CubicNorm() << endl;

    std::cout << "\n---------Benchmarks--------" << endl;
    auto Inverse_test = BenchmarkInverse(TEST_COUNT, TEST_SIZE);
    cout << fixed << setprecision(7) << "Inverse matrix: " << MUSEC_TO_SEC(Inverse_test.time1) << "s." << endl;
    cout << defaultfloat << "Min norm: " << Inverse_test.min_delta << ", Avg norm: " << Inverse_test.avg_delta << ", Max norm: " << Inverse_test.max_delta << endl << endl;

    auto Gauss_test = BenchmarkGauss(TEST_COUNT, TEST_SIZE);
    cout << fixed << setprecision(7) << "Gauss + Solve: " << MUSEC_TO_SEC(Gauss_test.time1) << "s." << endl;
    cout << defaultfloat << "Min norm: " << Gauss_test.min_delta << ", Avg norm: " << Gauss_test.avg_delta << ", Max norm: " << Gauss_test.max_delta << endl << endl;

    auto Relax_test = BenchmarkRelax(RELAX_EPS, TEST_COUNT, TEST_SIZE);
    cout << fixed << setprecision(7) << "Relaxation solve: " << MUSEC_TO_SEC(Relax_test.time1) << "s." << endl;
    cout << defaultfloat << "Min norm: " << Relax_test.min_delta << ", Avg norm: " << Relax_test.avg_delta << ", Max norm: " << Relax_test.max_delta << endl << endl;

    auto LUP_test = BenchmarkDecomposition(MatrixUtils::DecompositionLUP<double>, MatrixUtils::LinearSolveWithLUP<double>, TEST_COUNT, TEST_SIZE);
    auto LDLT_test = BenchmarkDecomposition(MatrixUtils::DecompositionLDLT<double>, MatrixUtils::LinearSolveWithLDLT<double>, TEST_COUNT, TEST_SIZE);
    
    std::cout << fixed << setprecision(7) << "LUP decompose: " << MUSEC_TO_SEC(LUP_test.time1) <<
                                          "s, LUP solve: " << MUSEC_TO_SEC(LUP_test.time2) << "s." << endl;
    cout << defaultfloat << "Min norm: " << LUP_test.min_delta << ", Avg norm: " << LUP_test.avg_delta << ", Max norm: " << LUP_test.max_delta << endl << endl;
    std::cout << fixed << setprecision(7) << "LDLt decompose: " << MUSEC_TO_SEC(LDLT_test.time1) <<
                                          "s, LDLt solve: " << MUSEC_TO_SEC(LDLT_test.time2) << "s." << endl;
    cout << defaultfloat << "Min norm: " << LDLT_test.min_delta << ", Avg norm: " << LDLT_test.avg_delta << ", Max norm: " << LDLT_test.max_delta << endl << endl;

    return 0;
};
