#pragma once

template <typename ValueType>
class LinAlEntity {
public:
	virtual void ChangeRow(size_t first, size_t second) = 0;
	virtual void MultiplyRow(size_t row, ValueType value) = 0;
	virtual void AddRowMultiplied(size_t first, size_t second, ValueType value) = 0;
};

