#pragma once
#include "LinAlEntity.h"
#include <stdlib.h>
#include <iostream>
#include <new>
#include <cstring>
#include <iomanip>

template <typename T>
class Vector {
	template<typename T>
	friend class Matrix;
	friend class MatrixUtils;
	friend class GaussHelper;
	friend class LinearSolve;
	friend class Decompositions;

public:
#pragma region ctor/dtor
	Vector(size_t size) : size_(size) {
		AllocateMemory(size);
	}

	Vector(const Vector& vector) {
		Copy(vector);
	}

	Vector& operator=(const Vector& vector) {
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
	void print() {
		for (size_t i = 0; i < size_; i++) {
			std::cout << std::setw(6) << std::setprecision(3) << data[i] << std::endl;
		}
	}
#pragma endregion

private:
	T* data;
	size_t size_;

private:
#pragma region ctor/dtor helpers
	void Copy(const Vector& vector) {
		//std::cout << "Copied vector\n";
		size_ = vector.size_;
		AllocateMemory(size_);
		memcpy(data, vector.data, sizeof(T) * size_);
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
