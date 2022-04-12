#pragma once

template <typename T>
class LinAlEntity {
public:
	virtual void ChangeRow(size_t first, size_t second) = 0;
	virtual void MultiplyRow(size_t row, T value) = 0;
	virtual void AddRowMultiplied(size_t first, size_t second, T value) = 0;
};

