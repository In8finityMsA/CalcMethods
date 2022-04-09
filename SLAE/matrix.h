#pragma once
#include <stdlib.h>
#include <iostream>
#include <new>
#include <cstring>
#include <iomanip>
#include "vector.h"

template <typename ValueType>
class Matrix {

public:
	Matrix(size_t size) : size_(size) {
		AllocateMemory(size);
	}

	Matrix(Matrix&& matrix) noexcept {
		Move(std::forward<Matrix>(matrix));
	}

	Matrix& operator=(Matrix&& matrix) noexcept {
		FreeMemory();
		Move(std::forward<Matrix>(matrix));
	}

	Matrix(const Matrix& matrix) {
		Copy(matrix);
	}

	Matrix& operator=(const Matrix& matrix) {
		FreeMemory();
		Copy(matrix);
	}

	virtual ~Matrix() {
		FreeMemory();
	}


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

	size_t size() {
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
	
	void print() {
		for (size_t i = 0; i < size_; i++) {
			for (size_t j = 0; j < size_; j++) {
				std::cout << std::setw(6) << std::setprecision(3) << data[i][j] << "  ";
			}
			std::cout << std::endl;
		}
	}

private:
	ValueType** data;
	size_t size_;

private:
	void Copy(const Matrix& matrix) {
		std::cout << "very very bad\n";
		size_ = matrix.size_;
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
};

