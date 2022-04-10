#pragma once
#include <vector>
#include <chrono>
#include <functional>
#include <algorithm>

class Benchmark {

public:
	typedef std::chrono::microseconds duration;

	Benchmark(size_t times_repeat) : times_repeat(times_repeat), time(times_repeat) {
	}

	template<typename T>
	std::vector<duration> StartBenchmark(T binded_func) {
		for (size_t i = 0; i < times_repeat; i++) {
			auto start = std::chrono::high_resolution_clock::now();
			auto r = binded_func();
			auto end = std::chrono::high_resolution_clock::now();
			memcpy(trash, &r, sizeof(r));
			trash[rand() % sizeof(trash)] << 5; // avoiding optimizing out result
			time[i] = std::chrono::duration_cast<duration>(end - start);
		}
		return time;
	}

	void SetTimesRepeat(size_t times_repeat) {
		this->times_repeat = times_repeat;
		time = std::vector<duration>(times_repeat);
	}

	size_t GetTimesRepeat() const {
		return times_repeat;
	}

	std::vector<duration> GetLastBenchResult() const {
		return time;
	}

private:
	size_t times_repeat;
	std::vector<duration> time;
	unsigned char trash[2048];
};
