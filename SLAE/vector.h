#pragma once
#include "LinAlEntity.h"
#include <stdlib.h>
#include <iostream>
#include <new>
#include <cstring>

template <typename ValueType>
class Vector : public LinAlEntity<ValueType> {
	template<typename ValueType>
	friend class Matrix;
	friend class MatrixUtils;

public:
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

	size_t size() {
		return size_;
	}

	ValueType& operator()(int index) {
		return data[index];
	}

	const ValueType& operator()(int index) const {
		return data[index];
	}

	void ChangeRow(size_t first, size_t second) override {
		std::swap(data[first], data[second]);
	}
	void MultiplyRow(size_t row, ValueType value) override {
		data[row] *= value;
	}
	void AddRowMultiplied(size_t first, size_t second, ValueType value) override {
		data[second] += data[first] * value;
	}

	void print() {
		for (size_t i = 0; i < size_; i++) {
			std::cout << std::setw(6) << std::setprecision(3) << data[i] << std::endl;
		}
	}

private:
	ValueType* data;
	size_t size_;

private:
	void Copy(const Vector& vector) {
		std::cout << "Copied vector\n";
		size_ = vector.size_;
		AllocateMemory(size_);
		memcpy(data, vector.data, sizeof(ValueType) * size_);
	}

	void Move(Vector&& vector) noexcept {
		data = vector.data;
		size_ = vector.size_;

		vector.size_ = 0;
		vector.data = nullptr;
	}

	void AllocateMemory(size_t size) {
		data = new ValueType[size]{};
	}

	void FreeMemory() noexcept {
		delete[] data;
	}

};

#pragma once
