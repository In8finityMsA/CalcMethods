#pragma once
#include "matrix.h"
#include "vector.h"
#include <complex>
#include <vector>
#include <optional>

class EigenValues {
public:
	typedef std::complex<double> Complex;

	template<typename T>
	struct PowerMethodResult {
		T eig_val;
		std::optional<T> eig_val2;
		Vector<T> eig_vec;
		std::optional<Vector<T>> eig_vec2;
		size_t iterations;
	};

	static PowerMethodResult<Complex> PowerMethod(Matrix<double> m, double eps) {
		eps = std::max(1e-14, eps);
		size_t iteration = 0;
		const size_t b_size = 4;
		std::vector<Vector<double>> batch_vec( b_size , Vector<double>{m.size_} );
		std::vector<size_t> batch_ind( b_size , 0 );
		for (size_t i = 0; i < m.size_; i++) {
			batch_vec[0].data[i] = -5 +rand() % 10;
		}

		std::pair<Complex, Complex> eig_values{ {},{} };
		std::pair<Complex, Complex> eig_values_old{ {},{} };
		bool check_loop = false;
		while (iteration < 200000) {
			++iteration;

			for (size_t i = 1; i < b_size; i++) {
				batch_vec[i] = m * batch_vec[i - 1];
			}

			auto coefs = LinearLeastSquares(batch_vec[b_size - 3], batch_vec[b_size - 2], batch_vec[b_size - 1]);
			eig_values = SolveQuadratic(1, coefs.first, coefs.second);

			double delta1 = abs(abs(eig_values.first) - abs(eig_values_old.first));
			double delta2 = abs(abs(eig_values.second) - abs(eig_values_old.second));
			if (delta1 < eps && delta2 < eps) {
				return PowerResult(eig_values, batch_vec[b_size - 1], batch_vec[b_size - 2], iteration, eps);
			}

			eig_values_old = eig_values;
			batch_vec[b_size - 1].NormalizeEuclidean();
			batch_vec[0] = batch_vec[b_size - 1];
		}
		return PowerResult(eig_values, batch_vec[b_size - 1], batch_vec[b_size - 2], iteration, eps);
	}

	static PowerMethodResult<Complex> PowerResult(std::pair<Complex, Complex> eig_values, Vector<double> cur_vector, Vector<double> prev_vector, size_t iterations, double eps) {
		size_t size = cur_vector.size_;
		Vector<Complex> eig_vec1{ size };
		Vector<Complex> eig_vec2{ size };

		if (eig_values.first.imag() != 0 && eig_values.second.imag() != 0) {
			for (size_t i = 0; i < size; i++) {
				auto eig_val = eig_values.first;
				eig_vec1.data[i] = { cur_vector.data[i] - prev_vector.data[i] * eig_values.first.real(), prev_vector.data[i] * eig_values.first.imag() };
				eig_vec2.data[i] = { cur_vector.data[i] - prev_vector.data[i] * eig_values.second.real(), prev_vector.data[i] * eig_values.second.imag() };
			}
			eig_vec1.NormalizeCubic();
			eig_vec2.NormalizeCubic();
			return { eig_values.first, eig_values.second, eig_vec1, eig_vec2, iterations };

		} else if (abs(abs(eig_values.first) - abs(eig_values.second)) < 10*eps) {
			for (size_t i = 0; i < size; i++) {
				eig_vec1.data[i] = cur_vector.data[i] + eig_values.first * prev_vector.data[i];
				eig_vec2.data[i] = cur_vector.data[i] + eig_values.second * prev_vector.data[i];
			}
			eig_vec1.NormalizeCubic();
			eig_vec2.NormalizeCubic();
			return { eig_values.first, eig_values.second, eig_vec1, eig_vec2, iterations };
		} else {
			cur_vector.NormalizeCubic();
			return { eig_values.first, std::nullopt, cur_vector, std::nullopt, iterations };
		}
	}

	static void ReflectVector(Vector<double>& v, Vector<double>& w) {
		double scalar_mult = 0;
		for (size_t i = 0; i < w.size_; i++) {
			scalar_mult += v.data[i] * w.data[i];
		}
		for (size_t i = 0; i < w.size_; i++) {
			v.data[i] -= 2 * scalar_mult * w.data[i];
		}
	}

	static std::pair< Complex, Complex> SolveQuadratic(double a, double b, double c) {
		double D = b * b - 4 * a * c;
		double A = 2 * a;
		if (D < 0) {
			return { {-b / A, sqrt(-D) / A}, {-b / A, -sqrt(-D) / A } };
		} else {
			return { { (-b + sqrt(D)) / A, 0}, { (-b - sqrt(D)) / A, 0 } };
		}
	}

	static std::pair<double, double> LinearLeastSquares(Vector<double> u, Vector<double> Au, Vector<double> A2u) {
		auto norm = u.EuñlideanNorm();
		Vector<double> w(u);
		norm = copysign(norm, -w.data[0]); // sign of a norm is opposite to a_0 to avoid substracting close values
		w.data[0] -= norm;
		w.NormalizeEuclidean(); // w = u - u' / ||u - u'||_2
		u.data[0] = norm; // a' = {||a||_2, 0...}
		for (size_t i = 1; i < u.size_; i++) {
			u.data[i] = 0;
		}
		ReflectVector(Au, w);
		ReflectVector(A2u, w);

		norm = 0;
		for (size_t i = 1; i < Au.size_; i++) {
			norm += Au.data[i] * Au.data[i];
		}
		norm = copysign(sqrt(norm), -w.data[1]); // sign of a norm is opposite to a_1 to avoid substracting close values
		w = Au;
		w.data[0] = 0;
		w.data[1] -= norm;
		w.NormalizeEuclidean(); // w = u - u' / ||u - u'||_2
		Au.data[1] = norm; // a' = {||a||_2, 0...}
		for (size_t i = 2; i < Au.size_; i++) {
			Au.data[i] = 0;
		}
		ReflectVector(A2u, w);

		double c1 = -A2u.data[1] / Au.data[1]; // minus for A2u because we solve ||{u,Au}x + A2u|| -> min
		double c0 = (-A2u.data[0] - Au.data[0]*c1) / u.data[0];
		return { c1, c0 };
	}

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
				if (d >= 0) {
					return {}; // Need more iterations
				}
				double sqrt_d = sqrt(-d);

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