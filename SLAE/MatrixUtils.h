#pragma once
#include "matrix.h"
#include "vector.h"
#include <vector>
#include <memory>
#include <cmath>


class MatrixUtils {
public:
	enum class GaussForm {
		TRIANGLE,
		DIAGONAL,
	};
	enum class LeadingChoice {
		NO_CHOICE,
		COLUMN,
		ROW,
		MATRIX
	};

	template<typename T>
	struct LUP_Dense {
		Matrix<T> LU;
		std::vector<size_t> p;
	};

	template<typename T>
	struct LUP {
		LUP(size_t s) {
			L = Matrix<T>::GetIdentity(s);
			U = Matrix<T>(s);
			P = Matrix<T>(s);
		}

		Matrix<T> L;
		Matrix<T> U;
		Matrix<T> P;
	};

	template<typename T>
	struct LDLT_Dense {
		Matrix<T> LLT;
		std::vector<int> d;
	};

	template<typename T>
	struct LDLT {
		LDLT(size_t s) {
			L = Matrix<T>(s);
			D = Matrix<T>(s);
			LT = Matrix<T>(s);
		}

		Matrix<T> L;
		Matrix<T> D;
		Matrix<T> LT;
	};

#pragma region lin_solve
	template<typename T>
	static Vector<T> LinearSolve(Matrix<T> a, Vector<T> b) {
		Gauss(a, b, GaussForm::TRIANGLE, LeadingChoice::COLUMN);
		return SolveUpperTriangle(a, b);
	}

	template<typename T>
	static Vector<T> LinearSolveWithLUP(const LUP_Dense<T>& dense, Vector<T> b) {
		// Permute vector b
		Permutation(b, dense.p);
		// Pass true to make it ignore diagonal values and use ones instead
		auto y = SolveLowerTriangle(dense.LU, b, true);
		return SolveUpperTriangle(dense.LU, y);
	}

	template<typename T>
	static Vector<T> LinearSolveWithLDLT(LDLT_Dense<T>& dense, const Vector<T>& b) {
		auto y = SolveLowerTriangle(dense.LLT, b, false);
		for (size_t i = 0; i < dense.LLT.size_; i ++) {
			dense.LLT.data[i][i] *= dense.d[i];
		}
		auto res = SolveUpperTriangle(dense.LLT, y);

		// return LT to initial form
		for (size_t i = 0; i < dense.LLT.size_; i++) {
			dense.LLT.data[i][i] *= dense.d[i];
		}
		return res;
	}
#pragma endregion

#pragma region decomposition
	template<typename T>
	static LUP_Dense<T> DecompositionLUP(Matrix<T> a) {
		auto p = GaussLU(a);
		return { a, p };
	}

	template<typename T>
	static LUP<T> UndenseLUP(LUP_Dense<T> dense) {
		size_t s = dense.LU.size_;
		LUP<T> lup{ s };
		
		for (size_t i = 0; i < s; i++) {
			lup.U.data[i][i] = dense.LU.data[i][i];
			for (size_t j = 0; j < i; j++) {
				lup.L.data[i][j] = dense.LU.data[i][j];
				lup.U.data[s-1-i][s-1-j] = dense.LU.data[s-1-i][s-1-j];
			}
		}

		for (size_t i = 0; i < s; i++) {
			lup.P.data[i][dense.p[i]] = 1;
		}
		return lup;
	}

	template<typename T>
	static LDLT_Dense<T> DecompositionLDLT(Matrix<T> a) {
		GaussLDLT(a);

		size_t s = a.size_;
		std::vector<int> d(s, 1);
		for (size_t i = 0; i < s; i++) {
			T divisor = sqrt(fabs(a[i][i]));
			int sign = a[i][i] >= 0 ? 1 : -1;
			d[i] = sign;

			a.data[i][i] /= divisor;
			for (size_t j = i + 1; j < s; j++) {
				a.data[j][i] /= divisor;
				a.data[i][j] = a.data[j][i];
				a.data[i][j] *= sign;
			}
		}
		return { a, d };
	}

	template<typename T>
	static LDLT<T> UndenseLDLT(LDLT_Dense<T> dense) {
		size_t s = dense.LLT.size_;
		LDLT<T> ldlt{ s };

		for (size_t i = 0; i < s; i++) {
			ldlt.L.data[i][i] = dense.LLT.data[i][i] * dense.d[i];
			ldlt.LT.data[i][i] = ldlt.L.data[i][i];
			ldlt.D.data[i][i] = dense.d[i];
			for (size_t j = 0; j < i; j++) {
				ldlt.L.data[i][j] = dense.LLT.data[i][j] * dense.d[j];
				ldlt.LT.data[s - 1 - i][s - 1 - j] = dense.LLT.data[s - 1 - i][s - 1 - j];
			}
		}
		return ldlt;
	}
#pragma endregion

#pragma region iterations
	template<typename T>
	static Vector<T> Relaxation(Matrix<T> m, Vector<T> b, T relax_param, T eps) noexcept {
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
		} while (((x_cur - x_prev).CubicNorm() >= eps));
		return x_cur;
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
	static void Gauss(Matrix<T>& m, Vector<T>& ext, GaussForm gf, LeadingChoice lc) noexcept {
		for (size_t iter = 0; iter < m.size_; iter++) {
			if (lc == LeadingChoice::COLUMN) {
				auto max_index = ChooseByColumn(m, iter);
				if (max_index != iter) {
					// Change current row and row with max element
					std::swap(m.data[iter], m.data[max_index]);
					std::swap(ext.data[iter], ext.data[max_index]);
				}
			}
			// If making diagonal we need to add row to all the other rows. If triangle - only to rows under current
			size_t i = gf == GaussForm::DIAGONAL ? 0 : iter + 1; 
			// Add current row to all under it
			for (; i < m.size_; i++) {
				if (i == iter) continue;

				T multiplier = -m.data[i][iter] / m.data[iter][iter];
				ext.data[i] += ext.data[iter] * multiplier;
				for (size_t j = iter; j < m.size_; j++) {
					m.data[i][j] += m.data[iter][j] * multiplier;
				}
			}
		}
	}

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

	template<typename T> // Storing in lower triangle
	static void GaussLDLT(Matrix<T>& m) noexcept {
		for (size_t iter = 0; iter < m.size_ - 1; iter++) {
			// Add current row to all under it
			for (size_t i = iter + 1; i < m.size_; i++) {

				T multiplier = -m.data[i][iter] / m.data[iter][iter];
				for (size_t j = iter + 1; j <= i; j++) {
					m.data[i][j] += m.data[j][iter] * multiplier;
				}
			}
		}
	}

	template<typename T>
	static std::vector<size_t> GaussLU(Matrix<T>& lu) noexcept {
		std::vector<size_t> permut;
		permut.reserve(lu.size_);
		for (size_t i = 0; i < lu.size_; i++) {
			permut.push_back(i);
		}

		for (size_t iter = 0; iter < lu.size_ - 1; iter++) {
			auto max_index = ChooseByColumn(lu, iter);
			if (max_index != iter) {
				// Change current row and row with max element
				std::swap(lu.data[iter], lu.data[max_index]);
				// Change rows in P
				std::swap(permut[iter], permut[max_index]);
			}

			// Add current row to all under it
			for (size_t i = iter + 1; i < lu.size_; i++) {

				T multiplier = -lu.data[i][iter] / lu.data[iter][iter];
				for (size_t j = iter; j < lu.size_; j++) {
					lu.data[i][j] += lu.data[iter][j] * multiplier;
				}
				lu[i][iter] = -multiplier;
			}
		}
		return permut;
	}

	template<typename T>
	inline static size_t ChooseByColumn(Matrix<T>& m, size_t iter) noexcept {
		T max_elem = fabs(m.data[iter][iter]);
		size_t max_index = iter;
		for (size_t i = iter; i < m.size_; i++) {
			if (std::abs(m.data[i][iter]) > max_elem) {
				max_elem = fabs(m.data[i][iter]);
				max_index = i;
			}
		}
		return max_index;
	}
#pragma endregion 

	template<typename T>
	static Vector<T> SolveUpperTriangle(const Matrix<T>& m, const Vector<T>& ext) {
		Vector<T> answer{ ext.size_ };
		for (size_t i = m.size_ - 1; i < SIZE_MAX; i--) {
			T sum = 0;
			for (size_t j = i + 1; j < m.size_; j++) {
				sum += m.data[i][j] * answer.data[j];
			}
			answer.data[i] = (-sum + ext.data[i]) / m.data[i][i];
		}
		return answer;
	}

	template<typename T>
	static Vector<T> SolveLowerTriangle(const Matrix<T>& m, const Vector<T>& ext, bool identity_diag) {
		Vector<T> answer{ ext.size_ };
		for (size_t i = 0; i < m.size_; i++) {
			T sum = 0;
			for (size_t j = i - 1; j < SIZE_MAX; j--) {
				sum += m.data[i][j] * answer.data[j];
			}

			answer.data[i] = (-sum + ext.data[i]);
			if (!identity_diag) {
				answer.data[i] /= m.data[i][i];
			}
		}
		return answer;
	}

	template<typename T>
	static void Permutation(Vector<T>& A, std::vector<size_t> P) {
		// For each element of P
		size_t n = P.size();
		for (int i = 0; i < n; i++) {
			int next = i;

			// Check if it is already
			// considered in cycle
			while (P[next] != SIZE_MAX) {

				// Swap the current element according
				// to the permutation in P
				std::swap(A(i), A(P[next]));
				int temp = P[next];

				// Subtract n from an entry in P
				// to make it negative which indicates
				// the corresponding move
				// has been performed
				P[next] = SIZE_MAX;
				next = temp;
			}
		}
	}
};

