#pragma once
#include <stdlib.h>
#include <iostream>
#include <new>
#include <cstring>
#include <iomanip>
//#include <algorithm>
#include <cmath>
#include "vector.h"
#include "LinAlEntity.h"

#define epsilon 10e-15

template <typename ValueType>
class Matrix : public LinAlEntity<ValueType> {
	friend class MatrixUtils;

public:
#pragma region ctr/dtr
	Matrix(size_t size) : size_(size) {
		AllocateMemory(size);
	}

	Matrix(const Matrix& matrix) {
		Copy(matrix);
	}

	Matrix& operator=(const Matrix& matrix) {
		FreeMemory();
		Copy(matrix);
		return *this;
	}

	Matrix(Matrix&& matrix) noexcept {
		Move(std::forward<Matrix>(matrix));
	}

	Matrix& operator=(Matrix&& matrix) noexcept {
		FreeMemory();
		Move(std::forward<Matrix>(matrix));
		return *this;
	}

	virtual ~Matrix() {
		FreeMemory();
	}
#pragma endregion

#pragma region arithmetics
	Matrix<ValueType> operator*(const Matrix<ValueType>& rhs) {
		Matrix<ValueType> out{size_};
		for (size_t i = 0; i < size_; i++) {
			for (size_t j = 0; j < size_; j++) {
				for (size_t k = 0; k < size_; k++) {
					out.data[i][j] += data[i][k] * rhs.data[k][j];
				}
			}
		}
		return out;
	}

	Vector<ValueType> operator*(const Vector<ValueType>& rhs) {
		Vector<ValueType> out{ size_ };
		for (size_t i = 0; i < size_; i++) {
			for (size_t k = 0; k < size_; k++) {
				out.data[i] += data[i][k] * rhs.data[k];
			}
		}
		return out;
	}

	void Transpose() {
		for (size_t i = 0; i < size_; i++) {
			for (size_t j = i + 1; j < size_; j++) {
				std::swap(data[i][j], data[j][i]);
			}
		}
	}

	Matrix<ValueType> GetTranspose() const {
		Matrix<ValueType> t{ size_ };
		for (size_t i = 0; i < size_; i++) {
			for (size_t j = i; j < size_; j++) {
				t.data[j][i] = data[i][j];
			}
		}
		return t;
	}
#pragma endregion

#pragma region norms
	ValueType QuadraticNorm() {
		ValueType norm = 0;
		for (size_t i = 0; i < size_; i++) {
			ValueType sum = 0;
			for (size_t j = 0; j < size_; j++) {
				sum += data[i][j];
			}
			norm = sum > norm ? sum : norm;
		}
	}
#pragma endregion

#pragma region algo
	void ChangeRow(size_t first, size_t second) override {
		std::swap(data[first], data[second]);
	}
	void MultiplyRow(size_t row, ValueType value) override {
		for (size_t j = 0; j < size_; j++) {
			data[row][j] *= value;
		}
	}
	void AddRowMultiplied(size_t first, size_t second, ValueType value) override {
		for (size_t j = 0; j < size_; j++) {
			data[second][j] += data[first][j] * value;
		}
	}
	
#pragma endregion

#pragma region getters
	static Matrix<ValueType> GetIdentity(size_t size) {
		Matrix<ValueType> identity{ size };
		for (size_t i = 0; i < size; i++) {
			identity.data[i][i] = 1;
		}
		return identity;
	}

	size_t Size() const {
		return size_;
	}

	ValueType& operator()(int row, int column) {
		return data[row][column];
	}

	const ValueType& operator()(int row, int column) const {
		return data[row][column];
	}

	ValueType* operator[](int index) {
		return data[index];
	}

	const ValueType* operator[](int index) const {
		return data[index];
	}
#pragma endregion

#pragma region presentation
	void print() const {
		for (size_t i = 0; i < size_; i++) {
			for (size_t j = 0; j < size_; j++) {
				/*std::cout << std::setw(6) << std::setprecision(3) << (data[i][j] > epsilon ? data[i][j] : 0) << "  ";*/
				std::cout << std::setw(6) << std::setprecision(3) << (fabs(data[i][j]) > epsilon ? data[i][j] : 0) << "  ";
			}
			std::cout << std::endl;
		}
	}
#pragma endregion

private:
	ValueType** data;
	size_t size_;


private:
#pragma region ctr/dtr_helpers
	void Copy(const Matrix& matrix) {
		//std::cout << "Copied matrix\n";
		size_ = matrix.size_;
		AllocateMemory(size_);
		for (size_t i = 0; i < size_; i++) {
			memcpy(data[i], matrix.data[i], sizeof(ValueType) * size_);
		}
	}

	void Move(Matrix&& matrix) noexcept {
		data = matrix.data;
		size_ = matrix.size_;

		matrix.size_ = 0;
		matrix.data = nullptr;
	}

	void AllocateMemory(size_t size) {
		data = new ValueType * [size] {};
		for (size_t i = 0; i < size; i++) {
			data[i] = new ValueType[size]{};
		}
	}

	void FreeMemory() noexcept {
		if (data != nullptr) {
			for (size_t i = 0; i < size_; i++) {
				delete[] data[i];
			}
			delete[] data;
		}
	}
#pragma endregion

#pragma region algo_helpers

#pragma endregion 
};

