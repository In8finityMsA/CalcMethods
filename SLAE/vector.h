#pragma once
#include <stdlib.h>
#include <iostream>
#include <new>
#include <cstring>
#include <iomanip>
#include <vector>
#include <cmath>
#include <complex>

template <typename T>
class Vector {
	template<typename T>
	friend class Matrix;
	friend class MatrixUtils;
	friend class GaussHelper;
	friend class LinearSolve;
	friend class Decompositions;
	friend class EigenValues;

public:
#pragma region ctor/dtor
	explicit Vector(size_t size) : size_(size) {
		AllocateMemory(size);
	}

	Vector(std::vector<T> vec) : size_(vec.size()) {
		AllocateMemory(size_);
		for (size_t i = 0; i < size_; i++) {
			data[i] = vec[i];
		}
	}

	Vector(const Vector<T>& vector) {
		Copy(vector);
	}

	template<typename U>
	Vector(const Vector<U>& vector) {
		Copy(vector);
	}

	Vector<T>& operator=(const Vector<T>& vector) {
		FreeMemory();
		Copy(vector);
		return *this;
	}

	template<typename U>
	Vector<T>& operator=(const Vector<U>& vector) {
		FreeMemory();
		Copy(vector);
		return *this;
	}

	Vector(Vector&& vector) noexcept {
		Move(std::forward<Vector>(vector));
	}

	Vector& operator=(Vector&& vector) noexcept {
		FreeMemory();
		Move(std::forward<Vector>(vector));
		return *this;
	}

	virtual ~Vector() {
		FreeMemory();
	}
#pragma endregion

#pragma region arithmetic
	Vector<T>& operator += (const Vector<T>& other) {
		for (size_t i = 0; i < size_; i++) {
			data[i] += other.data[i];
		}
		return *this;
	}

	Vector<T> operator + (const Vector<T>& other) const {
		Vector<T> res(*this);
		for (size_t i = 0; i < size_; i++) {
			res.data[i] = data[i] + other.data[i];
		}
		return res;
	}

	Vector<T>& operator -= (const Vector<T>& other) {
		for (size_t i = 0; i < size_; i++) {
			data[i] -= other.data[i];
		}
		return *this;
	}

	Vector<T> operator - (const Vector<T>& other) const {
		Vector<T> res(size_);
		for (size_t i = 0; i < size_; i++) {
			res.data[i] = data[i] - other.data[i];
		}
		return res;
	}

	template<typename Scalar>
	Vector<T>& operator *= (Scalar other) {
		for (size_t i = 0; i < size_; i++) {
			data[i] *= other;
		}
		return *this;
	}

	template<typename Scalar>
	Vector<T> operator * (Scalar other) const {
		Vector<T> res(size_);
		for (size_t i = 0; i < size_; i++) {
			res.data[i] = data[i] * other;
		}
		return res;
	}

	template<typename Scalar>
	Vector<T>& operator /= (Scalar other) {
		for (size_t i = 0; i < size_; i++) {
			data[i] /= other;
		}
		return *this;
	}

	template<typename Scalar>
	Vector<T> operator / (Scalar other) const {
		Vector<T> res(size_);
		for (size_t i = 0; i < size_; i++) {
			res.data[i] = data[i] / other;
		}
		return res;
	}

	double operator * (const Vector<T>& other) const {
		T res = 0;
		for (size_t i = 0; i < size_; i++) {
			res += data[i] * other.data[i];
		}
		return res;
	}
#pragma endregion

#pragma region norms
	T CubicNorm() const {
		T norm = 0;
		for (size_t i = 0; i < size_; i++) {
			norm = std::abs(data[i]) > norm ? std::abs(data[i]) : norm;
		}
		return norm;
	}

	T OctahedronNorm() const {
		T norm = 0;
		for (size_t i = 0; i < size_; i++) {
			norm += std::abs(data[i]);
		}
		return norm;
	}

	T EuñlideanNorm() const {
		T norm = 0;
		for (size_t i = 0; i < size_; i++) {
			norm += data[i] * data[i];
		}
		return sqrt(norm);
	}

	size_t NormalizeCubic() {
		T norm = 0;
		size_t index = 0;
		for (size_t i = 0; i < size_; i++) {
			if (std::abs(data[i]) > std::abs(norm)) {
				norm = std::abs(data[i]);
				index = i;
			}
		}
		for (size_t i = 0; i < size_; i++) {
			data[i] /= norm;
		}
		return index;
	}

	T NormalizeEuclidean() {
		T norm = EuñlideanNorm();
		for (size_t i = 0; i < size_; i++) {
			data[i] /= norm;
		}
		return norm;
	}
#pragma endregion

#pragma region getters
	size_t Size() const {
		return size_;
	}

	T& operator()(int index) {
		return data[index];
	}

	const T& operator()(int index) const {
		return data[index];
	}
#pragma endregion

#pragma region presentation
	void print(size_t precision = 3) {
		for (size_t i = 0; i < size_; i++) {
			std::cout << std::setw(3+precision) << std::setprecision(precision) << data[i] << std::endl;
		}
	}
#pragma endregion

private:
	T* data;
	size_t size_;

private:
#pragma region ctor/dtor helpers
	void Copy(const Vector<T>& vector) {
		//std::cout << "Copied vector\n";
		size_ = vector.size_;
		AllocateMemory(size_);
		memcpy(data, vector.data, sizeof(T) * size_);
	}

	template<typename U>
	void Copy(const Vector<U>& vector) {
		//std::cout << "Copied vector\n";
		size_ = vector.Size();
		AllocateMemory(size_);
		for (size_t i = 0; i < size_; i++) {
			data[i] = vector(i);
		}
	}

	void Move(Vector&& vector) noexcept {
		data = vector.data;
		size_ = vector.size_;

		vector.size_ = 0;
		vector.data = nullptr;
	}

	void AllocateMemory(size_t size) {
		data = new T[size]{};
	}

	void FreeMemory() noexcept {
		delete[] data;
	}
#pragma endregion

};

#pragma once
