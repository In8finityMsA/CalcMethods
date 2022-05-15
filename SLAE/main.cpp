#include <iostream>
#include "non_linear_solve.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include "vector.h"
#include "MatrixUtils.h"
#include "eigen_values.h"

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
	std::cout << "{ " << interval.first.left << ", " << interval.first.right << " }\n";
	auto root_newton = NonLinearSolve::NewtonMethod(func, func_derivative, interval.first, 1e-15);
	printf("Result newton: %.15f. Iterations: %zu\n", root_newton.first, root_newton.second);
	auto root_dichotomy = NonLinearSolve::DichotomyMethod(func, { 0, 2 }, 1e-15);
	printf("Result dichotomy: %.15f. Iterations: %zu\n", root_dichotomy.first, root_dichotomy.second);

	std::vector<std::vector<double>> A1_ = { {pow(N, 2) + 15, N - 1, -1, -2},
										   {N - 1, -15 - pow(N, 2), -N + 4, -4},
										   {-1, -N + 4, pow(N,2) + 8, -N},
										   {-2, -4, -N, pow(N,2) + 10} };

	Matrix<double> mat1(A1);
	auto hessenberg1 = MatrixUtils::GetHessenbergForm(mat1);
	//hessenberg1.print();

	Matrix<double> mat2(A2);
	auto hessenberg2 = MatrixUtils::GetHessenbergForm(mat2);
	hessenberg2.print();

	auto eig = EigenValues::QRalgorithm(hessenberg1, 1e-15);
	for (size_t i = 0; i < eig.size(); i++) {
		std::cout << eig[i] << '\n';
	}
	return 0;
}