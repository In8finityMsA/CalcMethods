#pragma once
#include "matrix.h"
#include "vector.h"

class GaussHelper {
public:
	enum class MatrixForm {
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
};