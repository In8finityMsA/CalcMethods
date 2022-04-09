#ifndef _MATRIX_GENERATOR_GUARD
#define _MATRIX_GENERATOR_GUARD
#include "matrix.h"
#include "vector.h"

Matrix<double> generateRealMatrix(size_t size, int variant);

Vector<double> generateRealVector(size_t size, int variant);

#endif //_MATRIX_GENERATOR_GUARD