#include "matrix_generator.h"
#include <random>
#include <cmath>
#include <chrono>

double getLowerBound(int variant) {
	return -std::exp2(variant / 4);
}

double getUpperBound(int variant) {
	return std::exp2(variant / 4);
}

Matrix<double> generateRealMatrix(size_t size, int variant, unsigned int seed) {
	Matrix<double> m{ size };
	std::default_random_engine generator{ seed };
	std::uniform_real_distribution<double> dist{ getLowerBound(variant), getUpperBound(variant) };

	for (int i = 0; i < size; i++) {
		double sum = 0;
		m(i, i) += 1;
		for (int j = i + 1; j < size; j++) {
			m(i, j) = dist(generator);
			m(j, i) = m(i, j);

			m(i, i) += std::fabs(m(i, j)); // to achieve diagonal domination
			m(j, j) += std::fabs(m(j, i));
		}
	}

	return m;
}

Vector<double> generateRealVector(size_t size, int variant, unsigned int seed) {
	Vector<double> v{ size };
	std::default_random_engine generator{ seed };
	std::uniform_real_distribution<double> dist{ getLowerBound(variant), getUpperBound(variant) };

	for (int i = 0; i < size; i++) {
		v(i) = dist(generator);
	}

	return v;
}