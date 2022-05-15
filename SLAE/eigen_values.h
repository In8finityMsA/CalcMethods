#pragma once
#include "matrix.h"
#include "vector.h"
#include <complex>
#include <vector>

class EigenValues {
public:
	static std::vector<std::complex<double>> QRalgorithm(Matrix<double> m, double eps) {
		size_t size = m.size_;
		size_t min_zeroes = (size - 1) / 2;
		bool is_get_min_zeroes = false;
		std::vector<std::complex<double>> eig_vals_cur{};
		std::vector<std::complex<double>> eig_vals_prev{};

		size_t iteration = 0;
		while (iteration < 100000) {
			++iteration;
			Matrix<double> Q = Matrix<double>::GetIdentity(size);
			for (int iter = 0; iter < size - 1; iter++) { // Select an element to make zero
				double root = sqrt(m.data[iter][iter] * m.data[iter][iter] + m.data[iter + 1][iter] * m.data[iter + 1][iter]); // sqrt(a**2 + b**2)
				double c = m.data[iter][iter] / root;
				double s = m.data[iter + 1][iter] / root;
				RotateRow(m, c, s, iter, iter + 1);
				RotateColumn(Q, c, -s, iter, iter + 1); // Q is multiplied right side by (Q_i)T 
			}
			m = m * Q;

			if (is_get_min_zeroes) {
				eig_vals_cur = ExtractEigenValuesQr(m, eps);
				if (eig_vals_cur.size() == m.size_ && eig_vals_prev.size() == m.size_) {
					double delta = 0;
					for (size_t i = 0; i < eig_vals_cur.size(); i++) {
						std::max(delta, std::abs(std::abs(eig_vals_cur[i]) - std::abs(eig_vals_prev[i])));
					}
					if (delta < eps) {
						return eig_vals_cur;
					}
				}
				eig_vals_prev = std::move(eig_vals_cur);
			} else {
				size_t zeroes_count = 0;
				for (size_t i = 0; i < size - 1; i++) {
					if (std::abs(Q.data[i + 1][i]) < eps) {
						++zeroes_count;
					}
				}
				is_get_min_zeroes = zeroes_count >= min_zeroes ? true : false;
			}
		}
		return eig_vals_cur;
	}

	static void RotateRow(Matrix<double>& m, double c, double s, size_t master, size_t slave) {
		for (size_t i = 0; i < m.size_; i++) {
			double temp = m.data[master][i];
			m.data[master][i] = c * m.data[master][i] + s * m.data[slave][i];
			m.data[slave][i] = -s * temp + c * m.data[slave][i];
		}
	}

	static void RotateColumn(Matrix<double>& m, double c, double s, size_t master, size_t slave) {
		for (size_t i = 0; i < m.size_; i++) {
			double temp = m.data[i][master];
			m.data[i][master] = c * m.data[i][master] + -s * m.data[i][slave];
			m.data[i][slave] = s * temp + c * m.data[i][slave];
		}
	}

	static std::vector<std::complex<double>> ExtractEigenValuesQr(Matrix<double> m, double eps) {
		enum EigenValueType : int {
			REAL = 0,
			COMPLEX = 1,
			NOT_READY = 2
		};
		std::vector<int> value_type = std::vector<int>(m.size_, 0);
		
		for (size_t i = 0; i < m.size_ - 1; i++) {
			if (std::abs(m.data[i+1][i]) > eps) {
				++value_type[i];
				++value_type[i + 1];
				if (value_type[i] == NOT_READY) {
					return {}; // Need more iterations
				}
			}
		}

		std::vector<std::complex<double>> eigen_values;
		for (size_t i = 0; i < m.size_; i++) {
			if (value_type[i] == REAL) {
				eigen_values.push_back(m.data[i][i]);
			} else if (value_type[i] == COMPLEX && value_type[i+1] == COMPLEX) {
				double b = m.data[i][i] + m.data[i + 1][i + 1];
				double d = b * b - 4 * (m.data[i][i] * m.data[i + 1][i + 1] - m.data[i + 1][i] * m.data[i][i + 1]);
				double sqrt_d = d < 0 ? sqrt(-d) : 0;

				double real = b / 2;
				double complex = sqrt_d / 2;
				eigen_values.emplace_back(real, complex);
				eigen_values.emplace_back(real, -complex);
				++i; // skip one step, complex pair was found
			}
		}
		return eigen_values;
	}
};