#pragma once
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cmath>
#include "vector.h"

constexpr double epsilon = 10e-15;

template <typename T>
class Matrix {
	friend class MatrixUtils;
	friend class GaussHelper;
	friend class LinearSolve;
	friend class Decompositions;
	friend class EigenValues;

public:
#pragma region ctr/dtr
	Matrix() noexcept {}

	explicit Matrix(size_t size) : size_(size) {
		AllocateMemory(size);
	}

	Matrix(std::vector<std::vector<T>> vec) : size_(vec.size()) {
		AllocateMemory(size_);
		for (size_t i = 0; i < size_; i++) {
			for (size_t j = 0; j < size_; j++) {
				data[i][j] = vec[i][j];
			}
		}
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
	Matrix<T> operator+(const Matrix<T>& rhs) const {
		Matrix<T> out{ size_ };
		for (size_t i = 0; i < size_; i++) {
			for (size_t j = 0; j < size_; j++) {
				out.data[i][j] = data[i][j] + rhs.data[i][j];
			}
		}
		return out;
	}

	Matrix<T> operator-(const Matrix<T>& rhs) const {
		Matrix<T> out{ size_ };
		for (size_t i = 0; i < size_; i++) {
			for (size_t j = 0; j < size_; j++) {
				out.data[i][j] = data[i][j] - rhs.data[i][j];
			}
		}
		return out;
	}

	Matrix<T> operator*(const Matrix<T>& rhs) const {
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

	Vector<T> operator*(const Vector<T>& rhs) const {
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
			for (size_t j = 0; j < size_; j++) {
				t.data[j][i] = data[i][j];
			}
		}
		return t;
	}
#pragma endregion

#pragma region gauss
	void AddColumnsMultiplied(size_t from, size_t to, double mult) noexcept {
		for (int j = 0; j < size_; j++) {
			data[j][to] += data[j][from] * mult;
		}
	}

	void AddRowsMultiplied(size_t from, size_t to, double mult) noexcept {
		for (int j = 0; j < size_; j++) {
			data[to][j] += data[from][j] * mult;
		}
	}

	void ChangeColumns(size_t first, size_t second) noexcept {
		for (int j = 0; j < size_; j++) {
			std::swap(data[j][first], data[j][second]);
		}
	}

	void ChangeRows(size_t first, size_t second) noexcept {
		std::swap(data[first], data[second]);
	}
#pragma endregion 

#pragma region norms
	T CubicNorm() {
		T norm = 0;
		for (size_t i = 0; i < size_; i++) {
			T sum = 0;
			for (size_t j = 0; j < size_; j++) {
				sum += std::abs(data[i][j]);
			}
			norm = std::max(norm, sum);
		}
		return norm;
	}

	T OctahedronNorm() {
		T norm = 0;
		for (size_t i = 0; i < size_; i++) {
			T sum = 0;
			for (size_t j = 0; j < size_; j++) {
				sum += std::abs(data[j][i]);
			}
			norm = std::max(norm, sum);
		}
		return norm;
	}
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
				std::cout << std::setw(6) <<  std::setprecision(3) << (fabs(data[i][j]) > epsilon ? data[i][j] : 0) << "  ";
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
};

