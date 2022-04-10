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

template <typename T>
class Matrix {
	friend class MatrixUtils;

public:
#pragma region ctr/dtr
	Matrix() noexcept {}

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
	Matrix<T> operator*(const Matrix<T>& rhs) {
		Matrix<T> out{size_};
		for (size_t i = 0; i < size_; i++) {
			for (size_t j = 0; j < size_; j++) {
				for (size_t k = 0; k < size_; k++) {
					out.data[i][j] += data[i][k] * rhs.data[k][j];
				}
			}
		}
		return out;
	}

	Vector<T> operator*(const Vector<T>& rhs) {
		Vector<T> out{ size_ };
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

	Matrix<T> GetTranspose() const {
		Matrix<T> t{ size_ };
		for (size_t i = 0; i < size_; i++) {
			for (size_t j = i; j < size_; j++) {
				t.data[j][i] = data[i][j];
			}
		}
		return t;
	}
#pragma endregion

#pragma region norms
	T QuadraticNorm() {
		T norm = 0;
		for (size_t i = 0; i < size_; i++) {
			T sum = 0;
			for (size_t j = 0; j < size_; j++) {
				sum += std::abs(data[i][j]);
			}
			norm = sum > norm ? sum : norm;
		}
		return norm;
	}
#pragma endregion

#pragma region algo
	/*void ChangeRow(size_t first, size_t second) override {
		std::swap(data[first], data[second]);
	}
	void MultiplyRow(size_t row, T value) override {
		for (size_t j = 0; j < size_; j++) {
			data[row][j] *= value;
		}
	}
	void AddRowMultiplied(size_t first, size_t second, T value) override {
		for (size_t j = 0; j < size_; j++) {
			data[second][j] += data[first][j] * value;
		}
	}*/
#pragma endregion

#pragma region getters
	static Matrix<T> GetIdentity(size_t size) {
		Matrix<T> identity{ size };
		for (size_t i = 0; i < size; i++) {
			identity.data[i][i] = 1;
		}
		return identity;
	}

	size_t Size() const {
		return size_;
	}

	inline T& operator()(int row, int column) noexcept {
		return data[row][column];
	}

	inline const T& operator()(int row, int column) const noexcept {
		return data[row][column];
	}

	T* operator[](int index) {
		return data[index];
	}

	const T* operator[](int index) const {
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
	T** data = nullptr;
	size_t size_ = 0;


private:
#pragma region ctr/dtr_helpers
	void Copy(const Matrix& matrix) {
		//std::cout << "Copied matrix\n";
		size_ = matrix.size_;
		AllocateMemory(size_);
		for (size_t i = 0; i < size_; i++) {
			memcpy(data[i], matrix.data[i], sizeof(T) * size_);
		}
	}

	void Move(Matrix&& matrix) noexcept {
		data = matrix.data;
		size_ = matrix.size_;

		matrix.size_ = 0;
		matrix.data = nullptr;
	}

	void AllocateMemory(size_t size) {
		data = new T * [size] {};
		for (size_t i = 0; i < size; i++) {
			data[i] = new T[size]{};
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

