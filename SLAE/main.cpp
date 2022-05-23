#include <iostream>
#include "non_linear_solve.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include "vector.h"
#include "MatrixUtils.h"
#include "eigen_values.h"
#include <iomanip>

#define N 13

double func(double x) {
	return (pow(x, 9) + M_PI) * cos(log(x * x + 1)) * exp(-x * x) - x / 2022.0;
}

double func_derivative(double x) {
	double first = -2 * exp(-x * x) * x * (pow(x, 9) + M_PI) * sin(log(x * x + 1)) / (x * x + 1);
	double second = -2 * exp(-x * x) * x * (pow(x, 9) + M_PI) * cos(log(x * x + 1));
	double third = 9 * exp(-x * x) * pow(x, 8) * cos(log(x * x + 1));
	return first + second + third - 1 / 2022.0;
}

std::vector<std::vector<double>> A1 = {
	{1,-2,1,0,-1,1,-2,2,0,-2},
	{0,2,0,0,2,1,-1,-1,-1,-2},
	{0,1,0,-1,1,-1,0,-1,1,-1},
	{-2,-1,2,-1,0,0,0,0,1,0},
	{1,-2,0,1,0,-2,-1,0,2,2},
	{-2,-2,0,-2,0,1,1,-2,1,1},
	{-1,-2,-1,-1,-2,-1,-2,1,-1,2},
	{-2,1,2,-2,0,2,1,-1,-2,2},
	{0,1,0,1,1,-2,2,0,1,1},
	{0,0,2,-1,-1,0,-2,2,-1,-1} };

std::vector<std::vector<double>> A2 = {
{ -1, 1, -1, 0, -1, 0, -1, 1, 1, -1, 0, -1, -1, 1, 0, 0, 1, 1, 1, 1 },
{ -1, 0, -1, 1, -1, 0, 0, 0, 0, -1, 0, 0, -1, 1, 0, -1, 1, -1, -1, 0 },
{ 1, 0, -1, 1, 0, 1, -1, -1, -1, 0, -1, -1, 1, -1, 1, 1, -1, 1, -1, 0 },
{ -1, 1, 0, 0, -1, 0, 0, -1, 0, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 0 },
{ 1, 0, -1, 0, 0, -1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, -1, 0, 0, 1 },
{ 0, 0, 0, 0, -1, 1, 1, 0, 0, 1, 1, 0, -1, 0, 1, 1, 0, 1, 0, 0 },
{ -1, 0, 1, 1, 1, -1, -1, 0, -1, 1, -1, -1, -1, 0, -1, 0, 0, 0, -1, 1 },
{ 0, 0, -1, -1, 0, 1, 1, 1, 1, -1, 0, 0, -1, 1, 1, 1, 1, 0, 0, -1 },
{ 0, 0, 1, 1, 0, 1, 1, 0, 1, -1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1 },
{ 0, -1, 0, 0, 1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 1, -1, 0, 0, 1, 1 },
{ 1, -1, 1, -1, -1, -1, 1, 0, -1, 0, 1, 1, -1, 0, 1, 1, 1, 0, 0, 0 },
{ 0, 1, 0, 0, -1, 0, 1, 0, 1, 0, 0, 1, 1, -1, -1, 0, -1, 1, 1, -1 },
{ -1, -1, -1, -1, 0, 1, -1, 0, 0, -1, 0, 0, 0, 1, 1, 0, 0, 0, -1, 0 },
{ -1, 0, 1, 0, -1, 0, 0, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 0 },
{ 1, -1, 0, -1, -1, 0, -1, -1, 0, 0, 1, 0, 1, 1, -1, 1, 0, 0, -1, 0 },
{ -1, -1, 1, 0, -1, 1, 1, -1, 1, 0, 0, -1, 1, -1, -1, 0, 0, 1, 1, 1 },
{ 0, 0, -1, 0, 0, 0, 0, -1, 1, 1, 0, -1, 1, -1, 0, 0, 0, -1, -1, 1 },
{ -1, 0, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 0, -1, 0, -1 },
{ -1, 0, 1, 0, 0, 0, 0, -1, 1, -1, 1, -1, 0, -1, -1, 1, 0, 1, 0, 0 },
{ 0, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 0, -1, -1, 0, 1, 0, -1, -1 } };

int main() {
	auto interval = NonLinearSolve::DichotomyInterval(func, { 0, 2 }, 1e-4);
	std::cout << "{ " << interval.first.left << ", " << interval.first.right << " }. Iterations: " << interval.second << '\n';
	auto root_newton = NonLinearSolve::NewtonMethod(func, func_derivative, interval.first, 1e-15);
	printf("Result newton: %.15f. Iterations: %zu\n", root_newton.first, root_newton.second);
	auto root_newton2 = NonLinearSolve::NewtonMethod(func, interval.first, 1e-15, 1e-5);
	printf("Result newton without derivative: %.15f. Iterations: %zu\n", root_newton2.first, root_newton2.second);
	auto root_dichotomy = NonLinearSolve::DichotomyMethod(func, { 0, 2 }, 1e-15);
	printf("Result dichotomy: %.15f. Iterations: %zu\n\n", root_dichotomy.first, root_dichotomy.second);

	std::vector<std::vector<double>> A1_ = { {pow(N, 2) + 15, N - 1, -1, -2},
										   {N - 1, -15 - pow(N, 2), -N + 4, -4},
										   {-1, -N + 4, pow(N,2) + 8, -N},
										   {-2, -4, -N, pow(N,2) + 10} };
	Matrix<double> mat0(std::vector<std::vector<double>>({ {18, -8, -20}, {20, -10, -20}, {8, -8, -10} }));
	Matrix<double> mat1(A1);
	auto hessenberg1 = MatrixUtils::GetHessenbergForm(mat1);
	//hessenberg1.print();

	Matrix<double> mat2(A2);
	auto hessenberg2 = MatrixUtils::GetHessenbergForm(mat2);
	//hessenberg2.print();

	auto eig = EigenValues::QRalgorithm(hessenberg1, 1e-15);
	for (size_t i = 0; i < eig.size(); i++) {
		std::cout << std::setprecision(15) << eig[i] << '\n';
	}

	auto mat_cur = mat1;
	auto power = EigenValues::PowerMethod(mat_cur, 1e-15);
	std::cout << "\nEig value: " << power.eig_val << '\n';
	std::cout << "\nEig vector: " << '\n';
	power.eig_vec.print();
	std::cout << "\nResidual vector: " << '\n';

	Matrix<std::complex<double>> mat_compl{ mat_cur.Size() };
	for (size_t i = 0; i < mat_cur.Size(); i++) {
		for (size_t j = 0; j < mat_cur.Size(); j++) {
			mat_compl(i,j) = mat_cur(i,j);
		}
	}
	((mat_compl * power.eig_vec) - (power.eig_vec * power.eig_val)).print();

	if (power.eig_vec2.has_value()) {
		std::cout << "\nEig value2: " << power.eig_val2.value() << '\n';
		power.eig_vec2.value().print();
		std::cout << "\nResidual vector: " << '\n';
		((mat_compl * power.eig_vec) - (power.eig_vec * power.eig_val)).print();
	}
	return 0;
}