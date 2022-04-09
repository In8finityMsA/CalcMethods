#pragma once
#include <stdlib.h>
#include <iostream>
#include <new>
#include <cstring>

template <typename ValueType>
class Vector {
	template<typename ValueType>
	friend class Matrix;

public:
	Vector(size_t size) : size_(size) {
		data = new ValueType[size] {};
	}

	Vector(Vector&& vector) noexcept {
		data = vector.data;
		size_ = vector.size_;

		vector.size_ = 0;
		vector.data = nullptr;
	}

	Vector(const Vector& vector) noexcept {
		std::cout << "very very bad\n";
		size_ = vector.size_;
		memcpy(data, vector.data, sizeof(ValueType) * size_);
	}

	virtual ~Vector() {
		delete[] data;
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

	void print() {
		for (size_t i = 0; i < size_; i++) {
			std::cout << std::setw(6) << std::setprecision(3) << data[i] << std::endl;
		}
	}

private:
	ValueType* data;
	size_t size_;
};

#pragma once
