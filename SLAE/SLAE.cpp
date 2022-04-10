#include <iostream>
#include "matrix.h"
#include "matrix_generator.h"
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include "MatrixUtils.h"

#define VARIANT 13

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

    std::cout << "Mod mat A:\n";
    mat.print();

    auto vec_answer_lu = MatrixUtils::LinearSolveWithLU(mat, vec_b);
    vec_answer_lu.print();

    auto vec_answer_ldl = MatrixUtils::LinearSolveWithLDL(mat, vec_b);
    vec_answer_ldl.print();
    

    //std::cout << "\nInverse:\n";
    //mat.Inverse().print();
    //std::cout << std::endl;
    //(mat * mat.Inverse()).print();

    const size_t iters = 256;
    const size_t size = 256;
    std::vector<unsigned long long> time;
    std::vector<unsigned long long> delta;
    for (int i = 0; i < iters; i++) {
        auto mat = generateRealMatrix(size, VARIANT, i);
        auto start = std::chrono::high_resolution_clock::now();
        auto inv = MatrixUtils::Inverse(mat);

        auto end = std::chrono::high_resolution_clock::now();
        
        time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        //std::cout << "Inverse: " << time.back() << std::endl;
        //auto iden = mat * inv;
        //iden.print();
        //std::cout << "\n\n";

    }
    std::cout << "Delta: " <<  std::accumulate(delta.begin(), delta.end(), 0) << std::endl;
    std::cout << "Time: " << std::accumulate(time.begin(), time.end(), 0) << std::endl;


    return 0;
}