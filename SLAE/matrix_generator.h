#ifndef _MATRIX_GENERATOR_GUARD
#define _MATRIX_GENERATOR_GUARD
#include "matrix.h"
#include "vector.h"

Matrix<double> generateRealMatrix(size_t size, int variant, unsigned int seed = 0);

Vector<double> generateRealVector(size_t size, int variant, unsigned int seed = 0);

#endif //_MATRIX_GENERATOR_GUARD