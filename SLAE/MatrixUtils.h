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

	template<typename ValueType>
	static Vector<ValueType> LinearSolve(Matrix<ValueType> a, Vector<ValueType> b) {
		Gauss(a, b, GaussForm::TRIANGLE, LeadingChoice::COLUMN);
		return SolveUpperTriangle(a, b);
	}

	template<typename ValueType>
	static Vector<ValueType> LinearSolveWithLU(Matrix<ValueType> a, Vector<ValueType> b) {
		auto LP = GaussLU(a, Vector<ValueType>(b));
		LP.second.Transpose();
		b = LP.second * b;
		auto y = SolveLowerTriangle(LP.first, b);
		return SolveUpperTriangle(a, y);
	}


	template<typename ValueType>
	static Vector<ValueType> LinearSolveWithLDL(Matrix<ValueType> a, Vector<ValueType> b) {
		Vector<ValueType>b_copy (b);
		Gauss(a, b_copy, GaussForm::TRIANGLE, LeadingChoice::NO_CHOICE);
		std::vector<bool> sign_negative(a.size_, false);
		for (size_t i = 0; i < a.size_; i++) {
			ValueType divisor = sqrt(fabs(a[i][i]));
			if (a[i][i] < 0) {
				divisor *= -1;
				sign_negative[i] = true;
			}
			
			for (size_t j = i; j < a.size_; j++) {
				a.data[i][j] /= divisor;
			}
		}
		Matrix<ValueType> L = a.GetTranspose();
		for (size_t i = 0; i < a.size_; i++) {
			if (sign_negative[i]) {
				for (size_t j = i; j < a.size_; j++) {
					a.data[i][j] *= -1;
				}
			}
		}
		std::cout << "Matrix L:\n";
		(L).print();
		std::cout << "Matrix LT:\n";
		(a).print();
		std::cout << "Matrix A:\n";
		(L* a).print();

		auto y = SolveLowerTriangle(L, b);
		return SolveUpperTriangle(a, y);
	}

	template<typename ValueType>
	static Matrix<ValueType> Inverse(Matrix<ValueType> m) noexcept {
		Matrix<ValueType> inv = Matrix<ValueType>::GetIdentity(m.size_);
		Gauss(m, inv, GaussForm::DIAGONAL, LeadingChoice::COLUMN);

		// Divide by diagonal element to get identity (currently it is just diagonal)
		for (size_t i = 0; i < m.size_; i++) {
			ValueType divisor = m.data[i][i];
			for (size_t j = 0; j < m.size_; j++) {
				inv[i][j] /= divisor;
			}
		}

		return inv;
	}

private:
	template<typename ValueType>
	static void Gauss(Matrix<ValueType>& m, LinAlEntity<ValueType>& ext, GaussForm gf, LeadingChoice lc) noexcept {
		for (size_t iter = 0; iter < m.size_; iter++) {

			if (lc == LeadingChoice::COLUMN) {
				ChooseByColumn(m, ext, iter);
			}

			size_t i = gf == GaussForm::DIAGONAL ? 0 : iter;
			for (; i < m.size_; i++) {
				if (i == iter) continue;

				ValueType multiplier = -m.data[i][iter] / m.data[iter][iter];
				ext.AddRowMultiplied(iter, i, multiplier);
				for (size_t j = iter; j < m.size_; j++) {
					m.data[i][j] += m.data[iter][j] * multiplier;
				}
			}
		}
	}

	template<typename ValueType>
	static std::pair<Matrix<ValueType>, Matrix<ValueType>> GaussLU(Matrix<ValueType>& m, Vector<ValueType> ext) noexcept {
		Matrix<ValueType> L = Matrix<ValueType>::GetIdentity(m.size_);
		Matrix<ValueType> P = Matrix<ValueType>::GetIdentity(m.size_);
		//std::vector<size_t> permut;
		/*permut.reserve(m.size_);
		for (size_t i = 0; i < m.size_; i++) {
			permut.push_back(i);
		}*/

		for (size_t iter = 0; iter < m.size_; iter++) {

			auto max_index = ChooseByColumn(m, ext, iter);
			// Change rows in L and create P
			if (max_index != iter) {
				for (size_t j = 0; j < iter; j++) {
					std::swap(L[iter][j], L[max_index][j]);
				}
				P.ChangeRow(iter, max_index);
				//std::swap(permut[iter], permut[max_index]);
			}

			for (size_t i = iter; i < m.size_; i++) {
				if (i == iter) continue;

				ValueType multiplier = -m.data[i][iter] / m.data[iter][iter];
				L[i][iter] = -multiplier;
				ext.AddRowMultiplied(iter, i, multiplier);
				for (size_t j = iter; j < m.size_; j++) {
					m.data[i][j] += m.data[iter][j] * multiplier;
				}
			}
		}
		return {L, P};
	}

	template<typename ValueType>
	inline static ValueType ChooseByColumn(Matrix<ValueType>& m, LinAlEntity<ValueType>& ext, size_t iter) noexcept {
		ValueType max_elem = fabs(m.data[iter][iter]);
		size_t max_index = iter;
		for (size_t i = iter; i < m.size_; i++) {
			if (fabs(m.data[i][iter]) > max_elem) {
				max_elem = fabs(m.data[i][iter]);
				max_index = i;
			}
		}

		if (max_elem > 0) {
			// Change current row and row with max element
			m.ChangeRow(iter, max_index);
			ext.ChangeRow(iter, max_index);
		}
		return max_index;
	}

	template<typename ValueType>
	static Vector<ValueType> SolveUpperTriangle(Matrix<ValueType>& m, Vector<ValueType>& ext) {
		Vector<ValueType> answer{ ext.size_ };
		for (size_t i = m.size_ - 1; i < SIZE_MAX; i--) {
			ValueType sum = 0;
			for (size_t j = i + 1; j < m.size_; j++) {
				sum += m.data[i][j] * answer.data[j];
			}
			answer.data[i] = (-sum + ext.data[i]) / m.data[i][i];
		}
		return answer;
	}

	template<typename ValueType>
	static Vector<ValueType> SolveLowerTriangle(Matrix<ValueType>& m, Vector<ValueType>& ext) {
		Vector<ValueType> answer{ ext.size_ };
		for (size_t i = 0; i < m.size_; i++) {
			ValueType sum = 0;
			for (size_t j = i - 1; j < SIZE_MAX; j--) {
				sum += m.data[i][j] * answer.data[j];
			}
			answer.data[i] = (-sum + ext.data[i]) / m.data[i][i];
		}
		return answer;
	}
};

