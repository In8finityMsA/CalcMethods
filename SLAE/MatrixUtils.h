#pragma once
#include "matrix.h"
#include "vector.h"
#include <vector>
#include <memory>
#include <cmath>


class MatrixUtils {
public:

	//template<typename T>
	static Matrix<double> GetHessenbergForm(Matrix<double> m) {
		size_t size = m.size_;
		for (int iter = size - 2; iter >= 0; iter--) { // select column for base element
			// Add current column to all left to it
			for (int i = iter - 1; i >= 0; i--) { // select column left to current

				if (m.data[iter + 1][iter] == 0) {
					m.ChangeColumns(iter, i);
					m.ChangeRows(iter, i);
					continue;
				}
				double multiplier = -m.data[iter+1][i] / m.data[iter+1][iter];
				m.AddColumnsMultiplied(iter, i, multiplier);
				m.AddRowsMultiplied(i, iter, -multiplier);
			}
		}
		return m;
	}

	static std::vector<double> QRalgorithm(Matrix<double> m) {

	}

#pragma region iterations
	template <typename T>
	struct RelaxResult {
		Vector<T> answer;
		std::vector<T> deltas;
	};

	template<typename T>
	static RelaxResult<T> Relaxation(Matrix<T> m, Vector<T> b, T relax_param, T eps) noexcept {
		// Make matrix B = E - DA and vector g = Db
		size_t s = m.size_;
		for (size_t i = 0; i < s; i++) {
			T divisor = m.data[i][i];
			for (size_t j = 0; j < s; j++) {
				m.data[i][j] /= -divisor;
			}
			b.data[i] /= divisor;
			m.data[i][i] = 0;
		}

		Vector<T> x_cur{ s };
		std::vector<T> deltas;
		for (size_t i = 0; i < s; i++) {
			x_cur.data[i] = 1;
		}
		Vector<T> x_prev{ s };
		int iter = 1;
		do {
			x_prev = x_cur;
			for (size_t i = 0; i < s; i++) {
				T new_coord = b.data[i];
				for (size_t j = 0; j < s; j++) {
					new_coord += m.data[i][j] * x_cur.data[j];
				}
				new_coord *= relax_param;
				x_cur.data[i] = new_coord + (1 - relax_param) * x_cur.data[i];
			}
			deltas.push_back((x_cur - x_prev).CubicNorm());
		} while (deltas.back() >= eps);
		return { x_cur , deltas };
	}
#pragma endregion

	template<typename T>
	static Matrix<T> Inverse(Matrix<T> m) noexcept {
		Matrix<T> inv = Matrix<T>::GetIdentity(m.size_);
		GaussInverse(m, inv);

		// Divide by diagonal element to get identity (currently it is just diagonal)
		for (size_t i = 0; i < m.size_; i++) {
			T divisor = m.data[i][i];
			for (size_t j = 0; j < m.size_; j++) {
				inv[i][j] /= divisor;
			}
		}

		return inv;
	}

private:
#pragma region gauss

	template<typename T>
	static void GaussInverse(Matrix<T>& m, Matrix<T>& ext) noexcept {
		for (size_t iter = 0; iter < m.size_; iter++) {
			// Add current row to all under it
			for (size_t i = 0; i < m.size_; i++) {
				if (i == iter) continue;

				T multiplier = -m.data[i][iter] / m.data[iter][iter];
				// Modify all row's elements in extended matrix
				for (size_t j = 0; j < iter; j++) {
					ext.data[i][j] += ext.data[iter][j] * multiplier;
				}
				// Modify elements only after iter to avoid multiplication by zero for optimization
				for (size_t j = iter; j < m.size_; j++) {
					ext.data[i][j] += ext.data[iter][j] * multiplier;
					m.data[i][j] += m.data[iter][j] * multiplier;
				}
			}
		}
	}

#pragma endregion 

};

