#pragma once
#include "matrix.h"
#include "vector.h"
#include "gauss_helper.h"

class Decompositions {
public: 
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

	template<typename T>
	static LUP_Dense<T> GetLUP(Matrix<T> a) {
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
				lup.U.data[s - 1 - i][s - 1 - j] = dense.LU.data[s - 1 - i][s - 1 - j];
			}
		}

		for (size_t i = 0; i < s; i++) {
			lup.P.data[i][dense.p[i]] = 1;
		}
		return lup;
	}

	template<typename T>
	static LDLT_Dense<T> GetLDLT(Matrix<T> a) {
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

private:
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
			auto max_index = GaussHelper::ChooseByColumn(lu, iter);
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
};