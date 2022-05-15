#pragma once
#include "matrix.h"
#include "vector.h"
#include "matrix_decompositions.h"
#include "gauss_helper.h"

class LinearSolve {
public:
	template<typename T>
	static Vector<T> SolveGauss(Matrix<T> a, Vector<T> b) {
		GaussInPlace(a, b, GaussHelper::MatrixForm::TRIANGLE, GaussHelper::LeadingChoice::COLUMN);
		return SolveUpperTriangle(a, b);
	}

	template<typename T>
	static Vector<T> SolveWithLUP(const Decompositions::LUP_Dense<T>& dense, Vector<T> b) {
		auto p = TransposeP(dense.p); // Get inverse of a permutation matrix
		Permutation(b, p); // permute vector b with matrix P
		// Pass true to make it ignore diagonal values and use ones instead
		auto y = SolveLowerTriangle(dense.LU, b, true);
		return SolveUpperTriangle(dense.LU, y);
	}

	template<typename T>
	static Vector<T> SolveWithLDLT(Decompositions::LDLT_Dense<T>& dense, const Vector<T>& b) {
		auto y = SolveLowerTriangle(dense.LLT, b, false);
		for (size_t i = 0; i < dense.LLT.size_; i++) {
			dense.LLT.data[i][i] *= dense.d[i];
		}

		auto res = SolveUpperTriangle(dense.LLT, y);
		// return LT to initial form
		for (size_t i = 0; i < dense.LLT.size_; i++) {
			dense.LLT.data[i][i] *= dense.d[i];
		}
		return res;
	}

private:
	template<typename T>
	static void GaussInPlace(Matrix<T>& m, Vector<T>& ext, GaussHelper::MatrixForm gf, GaussHelper::LeadingChoice lc) noexcept {
		for (size_t iter = 0; iter < m.size_; iter++) {
			if (lc == GaussHelper::LeadingChoice::COLUMN) {
				auto max_index = GaussHelper::ChooseByColumn(m, iter);
				if (max_index != iter) {
					// Change current row and row with max element
					std::swap(m.data[iter], m.data[max_index]);
					std::swap(ext.data[iter], ext.data[max_index]);
				}
			}
			// If making diagonal we need to add row to all the other rows. If triangle - only to rows under current
			size_t i = gf == GaussHelper::MatrixForm::DIAGONAL ? 0 : iter + 1;
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

private:
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
		size_t n = P.size();
		for (int i = 0; i < n; i++) {
			int next = i;

			while (P[next] != SIZE_MAX) {
				std::swap(A(i), A(P[next]));
				int temp = P[next];

				P[next] = SIZE_MAX;
				next = temp;
			}
		}
	}

	static std::vector<size_t> TransposeP(const std::vector<size_t>& P) {
		std::vector<size_t> Pt(P.size());
		for (size_t i = 0; i < P.size(); i++) {
			Pt[P[i]] = i;
		}
		return Pt;
	}
};