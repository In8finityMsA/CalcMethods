#include <iostream>
#include "matrix.h"
#include "matrix_generator.h"

#define VARIANT 13

int main()
{
    //Matrix<double> m{100};
    const size_t n = 3;
    auto mat = generateRealMatrix(n, VARIANT);
    mat.print();

    std::cout << "\nVector y:\n";
    auto vec_y = generateRealVector(n, VARIANT);
    vec_y.print();

    std::cout << "\nVector b:\n";
    auto vec_b = mat * vec_y;
    vec_b.print();
    std::cout << "Hello World!\n";
    return 0;
}