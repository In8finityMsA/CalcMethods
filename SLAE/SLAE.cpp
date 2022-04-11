#include <iostream>
#include "matrix.h"
#include "matrix_generator.h"
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include "MatrixUtils.h"
#include "Benchmark.h"
#include <functional>

#define VARIANT 13
using namespace std;

template<typename Dec, typename Solv>
std::pair<double, double> BenchmarkDecomposition(Dec decompose, Solv lin_solve, size_t times_repeat, size_t size) {
    long long time_acc = 0;
    long long time_acc2 = 0;

    for (size_t i = 0; i < times_repeat; i++) {
        auto m = generateRealMatrix(size, VARIANT);
        auto vec_x = generateRealVector(size, VARIANT);
        auto vec_b = m * vec_x;

        auto start = std::chrono::high_resolution_clock::now();
        auto dense = decompose(m);
        auto end = std::chrono::high_resolution_clock::now();
        time_acc += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        auto start2 = std::chrono::high_resolution_clock::now();
        auto res = lin_solve(dense, vec_b);
        auto end2 = std::chrono::high_resolution_clock::now();
        time_acc2 += std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

        double delta = (res - vec_x).OctahedronNorm();
        if (delta > 0.001) {
            std::cout << "Delta warning\n";
        } 
    }
    return { time_acc / times_repeat, time_acc2 / times_repeat };
}

int main()
{
    const size_t n = 3;
    auto mat = generateRealMatrix(n, VARIANT);
    mat.print();

    std::cout << "\nVector y:\n";
    auto vec_y = generateRealVector(n, VARIANT);
    vec_y.print();

    std::cout << "\nVector b:\n";
    auto vec_b = mat * vec_y;
    vec_b.print();

    auto vec_answer = MatrixUtils::LinearSolve(mat, vec_b);
    vec_answer.print();




    std::cout << "Before LU mat A:\n";
    mat.print();

    
    auto lup_dense = MatrixUtils::DecompositionLU(mat);
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

    /*auto vec_relax = MatrixUtils::Relaxation(mat, vec_b, 1.0, 10e-3);
    std::cout << "\n**Relax answer:\n";
    vec_relax.print();*/


    std::cout << "\n**Before LDLT mat A:\n";
    mat.print();
    auto ldlt1_dense = MatrixUtils::DecompositionLDLT(mat);
    auto ldlt_dense = MatrixUtils::DecompositionLDLT2(mat);
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


    auto LU_test = BenchmarkDecomposition(MatrixUtils::DecompositionLDLT2<double>, MatrixUtils::LinearSolveWithLDLT<double>, 1000, 256);
    auto LDLT_test = BenchmarkDecomposition(MatrixUtils::DecompositionLDLT<double>, MatrixUtils::LinearSolveWithLDLT<double>, 1000, 256);
    std::cout << fixed << setprecision(7) << "LDLt2 decompose: " << LU_test.first / 1000000.0 <<
                                    "s, LU solve: " << LU_test.second / 1000000.0 << "s." << endl;
    std::cout << fixed << setprecision(7) << "LDLt decompose: " << LDLT_test.first / 1000000.0 <<
                                    "s, LDLt solve: " << LDLT_test.second / 1000000.0 << "s." << endl;

    //const size_t iters = 256;
    //const size_t size = 256;
    //std::vector<unsigned long long> time;
    //std::vector<unsigned long long> delta;
    //for (int i = 0; i < iters; i++) {
    //    auto mat = generateRealMatrix(size, VARIANT, i);
    //    auto start = std::chrono::high_resolution_clock::now();
    //    auto inv = MatrixUtils::Inverse(mat);

    //    auto end = std::chrono::high_resolution_clock::now();
    //    
    //    time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    //    //std::cout << "Inverse: " << time.back() << std::endl;
    //    //auto iden = mat * inv;
    //    //iden.print();
    //    //std::cout << "\n\n";

    //}
    //std::cout << "Delta: " <<  std::accumulate(delta.begin(), delta.end(), 0) << std::endl;
    //std::cout << "Time: " << std::accumulate(time.begin(), time.end(), 0) << std::endl;


    return 0;
}