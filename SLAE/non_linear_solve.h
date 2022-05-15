#pragma once
#include <stdexcept>
#include <iostream>

class NonLinearSolve {
public:
	typedef double (*Function)(double x);

	struct Interval {
		Interval(double a, double b) : left{ a }, right{ b } {
			if (left > right) {
				std::swap(left, right);
			}
		}

		double getLength() {
			return right - left;
		}

		double left;
		double right;
	};

	static std::pair<double, size_t> NewtonMethod(Function f, Function f_der, Interval root_interval, double eps) {
		double x_prev = 0;
		double x_cur = root_interval.left;
		size_t iteration = 0;
		do {
			x_prev = x_cur;
			x_cur = x_prev - f(x_prev) / f_der(x_prev);
			//printf("#%d  x: %.15f, f(x): %.15f\n", iteration+1, x_cur, f(x_cur));
			++iteration;
		} while (std::abs(x_cur - x_prev) > eps);
		return { x_cur, iteration };
	}

	static std::pair<double, size_t> DichotomyMethod(Function f, Interval root_interval, double eps) {
		size_t iteration = 0;
		while (root_interval.getLength() > 2*eps) {
			double median = (root_interval.left + root_interval.right) / 2.0;
			double value = f(median);
			if (value * f(root_interval.left) >= 0) {
				root_interval.left = median;
			} else {
				root_interval.right = median;
			}
			++iteration;
		}
		return { (root_interval.left + root_interval.right) / 2.0, iteration };
	}

	static std::pair<Interval, size_t> DichotomyInterval(Function f, Interval root_interval, double eps) {
		size_t iteration = 0;
		while (root_interval.getLength() > eps) {
			double median = (root_interval.left + root_interval.right) / 2.0;
			double value = f(median);
			if (value * f(root_interval.left) >= 0) {
				root_interval.left = median;
			} else {
				root_interval.right = median;
			}
		}
		return { root_interval, iteration };
	}

};