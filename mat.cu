//
// Created by xixuan on 4/3/17.
//
// Note: everything is column majored.
//

#include "mat.h"

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>

// cuda kernels
// Everything should be column majored.

__global__ void prod(	const double *left_vec,
						const double *right_vec,
						double *result,
						int length) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < length)
		result[index] = left_vec[index] * right_vec[index];

}

__global__ void sum(	const double *left_vec,
						const double *right_vec,
						double *result,
						int length) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < length)
		result[index] = left_vec[index] + right_vec[index];

}

__global__ void sub(	const double *left_vec,
						const double *right_vec,
						double *result,
						int length) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < length)
		result[index] = left_vec[index] - right_vec[index];

}

__global__ void div(	const double *left_vec,
						const double *right_vec,
						double *result,
						int length) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < length)
		if ((right_vec[index] * right_vec[index]) > _SMALL_NUM_BOT)
			result[index] = left_vec[index] / right_vec[index];
		else
			result[index] = 0.0;

}

__global__ void row_1st_diff(	const double *dev_mat,
								double *dev_result,
								int num_rows,
								int num_cols,
								double delta) {
	/*	take the following matrix for example
	 * 	num_rows: 6 num_cols: 4
	 * 	0	6	12	18
	 * 	1	7	13	19
	 * 	2	8	14	20
	 * 	3	9	15	21
	 * 	4	10	16	22
	 * 	5	11	17	23
	 *
	 * 	row_index = 14 % 6	21 % 6
	 * 	col_index = 14 / 6	21 / 6
	 *
	 * 	IDX(row_index, col_index + 1, num_rows) = index + num_rows
	 * 	IDX(row_index, col_index - 1, num_rows) = index - num_rows
	 */

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < num_rows * num_cols) {
		int col_index = index / num_rows;

		if (num_cols >= 5) {
			if (col_index == 0) {
				dev_result[index] = (dev_mat[index + num_rows] -
										dev_mat[index]) / delta;
			}
			else if (col_index == num_cols - 1) {
				dev_result[index] = (dev_mat[index] -
										dev_mat[index - num_rows]) / delta;
			}
			else if ((col_index == 1) || (col_index == num_cols - 2)) {
				dev_result[index] = (dev_mat[index + num_rows] -
										dev_mat[index - num_rows]) / (2.0 * delta);
			}
			else {
				dev_result[index] = (- 0.0833333 * dev_mat[index + 2 * num_rows]
									 + 0.6666666 * dev_mat[index + num_rows]
									 - 0.6666666 * dev_mat[index - num_rows]
									 + 0.0833333 * dev_mat[index - 2 * num_rows]) / delta;
			}
			return;
		}

		else if (num_cols >= 3) {
			if (col_index == 0) {
				dev_result[index] = (dev_mat[index + num_rows] -
										dev_mat[index]) / delta;
			}
			else if (col_index == num_cols - 1) {
				dev_result[index] = (dev_mat[index] -
										dev_mat[index - num_rows]) / delta;
			}
			else {
				dev_result[index] = (dev_mat[index + num_rows] -
										dev_mat[index - num_rows]) / (2.0 * delta);
			}
			return;
		}

		else if (num_cols == 2) {
			if (col_index == 0) {
				dev_result[index] = (dev_mat[index + num_rows] -
										dev_mat[index]) / delta;
			}
			else {
				dev_result[index] = (dev_mat[index] -
										dev_mat[index - num_rows]) / delta;
			}
			return;
		}

		else {
			dev_result[index] = 0.0;
		}

	}

}

__global__ void row_1st_ele_inc(	const double *dev_mat,
									double *dev_result,
									int num_rows,
									int num_cols) {
	/*	take the following matrix for example
	 * 	num_rows: 6 num_cols: 4
	 * 	0	6	12	18
	 * 	1	7	13	19
	 * 	2	8	14	20
	 * 	3	9	15	21
	 * 	4	10	16	22
	 * 	5	11	17	23
	 *
	 * 	row_index = 14 % 6	21 % 6
	 * 	col_index = 14 / 6	21 / 6
	 *
	 * 	IDX(row_index, col_index + 1, num_rows) = index + num_rows
	 * 	IDX(row_index, col_index - 1, num_rows) = index - num_rows
	 */

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < num_rows * num_cols) {
		int col_index = index / num_rows;

		if (num_cols > 1) {
			if (col_index < num_cols - 1)
				dev_result[index] = dev_mat[index + num_rows] - dev_mat[index];
			else
				dev_result[index] = 0.0;
			return;
		}
		else
			dev_result[index] = 0.0;
	}

}

__global__ void col_1st_diff(	const double *dev_mat,
								double *dev_result,
								int num_rows,
								int num_cols,
								double delta) {
	/*	take the following matrix for example
	 * 	num_rows: 6 num_cols: 4
	 * 	0	6	12	18
	 * 	1	7	13	19
	 * 	2	8	14	20
	 * 	3	9	15	21
	 * 	4	10	16	22
	 * 	5	11	17	23
	 *
	 * 	row_index = 14 % 6	21 % 6
	 * 	col_index = 14 / 6	21 / 6
	 *
	 * 	IDX(row_index + 1, col_index, num_rows) = index + 1
	 * 	IDX(row_index - 1, col_index, num_rows) = index - 1
	 */

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < num_rows * num_cols) {
		int row_index = index % num_rows;

		if (num_rows >= 5) {
			if (row_index == 0) {
				dev_result[index] = (dev_mat[index + 1] -
										dev_mat[index]) / delta;
			}
			else if (row_index == num_rows - 1) {
				dev_result[index] = (dev_mat[index] -
										dev_mat[index - 1]) / delta;
			}
			else if ((row_index == 1) || (row_index == num_rows - 2)) {
				dev_result[index] = (dev_mat[index + 1] -
										dev_mat[index - 1]) / (2.0 * delta);
			}
			else {
				dev_result[index] = (- 0.0833333 * dev_mat[index + 2]
									 + 0.6666666 * dev_mat[index + 1]
									 - 0.6666666 * dev_mat[index - 1]
									 + 0.0833333 * dev_mat[index - 2]) / delta;
			}
			return;
		}

		else if (num_rows >= 3) {
			if (row_index == 0) {
				dev_result[index] = (dev_mat[index + 1] -
										dev_mat[index]) / delta;
			}
			else if (row_index == num_rows - 1) {
				dev_result[index] = (dev_mat[index] -
										dev_mat[index - 1]) / delta;
			}
			else {
				dev_result[index] = (dev_mat[index + 1] -
										dev_mat[index - 1]) / (2.0 * delta);
			}
			return;
		}

		else if (num_rows == 2) {
			if (row_index == 0) {
				dev_result[index] = (dev_mat[index + 1] -
										dev_mat[index]) / delta;
			}
			else {
				dev_result[index] = (dev_mat[index] -
										dev_mat[index - 1]) / delta;
			}
			return;
		}

		else {
			dev_result[index] = 0.0;
		}

	}

}

__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int *address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);

}

__global__ void vertical_element_shift(	const double *dev_mat,
										const double *dev_vec,
										double *dev_result,
										int num_rows,
										int num_cols,
										double min_K,
										double delta_K) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < num_rows * num_cols) {
		int row_index = index % num_rows;
		int col_index = index / num_rows;

		double lower_row_index, higher_row_index;
		double from_this_row_index =
				(row_index + min_K / delta_K) / dev_vec[col_index] - min_K / delta_K;

		lower_row_index = std::floor(from_this_row_index);
		higher_row_index = std::ceil(from_this_row_index);

		if (lower_row_index < 0.0) {
			atomicAdd(dev_result + IDX(0, col_index, num_rows), dev_mat[index]);
		}
		else if (higher_row_index > (num_rows - 1.0)) {
			atomicAdd(dev_result + IDX(num_rows - 1, col_index, num_rows), dev_mat[index]);
		}
		else {
			if ((higher_row_index + lower_row_index) >= (2.0 * from_this_row_index))
				atomicAdd(dev_result + IDX((int) lower_row_index, col_index, num_rows), dev_mat[index]);
			else
				atomicAdd(dev_result + IDX((int) higher_row_index, col_index, num_rows), dev_mat[index]);
		}

	}

}

__global__ void colwise_sum(	const double *dev_mat,
								double *dev_result,
								int num_rows,
								int num_cols) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < num_cols) {
		dev_result[index] = 0.0;
		for (int i = 0; i < num_rows; i++) {
			dev_result[index] += dev_mat[IDX(i, index, num_rows)];
		}
	}
}

__global__ void colwise_div_by_num(	const double *dev_mat,
									const double *dev_vec,
									double *dev_result,
									int num_rows,
									int num_cols) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < num_rows * num_cols) {
		int col_index = index / num_rows;

		dev_result[index] = dev_mat[index] / dev_vec[col_index];

	}

}

__global__ void colwise_prod_by_num(	const double *dev_mat,
										const double *dev_vec,
										double *dev_result,
										int num_rows,
										int num_cols) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < num_rows * num_cols) {
		int col_index = index / num_rows;

		dev_result[index] = dev_mat[index] * dev_vec[col_index];

	}

}

__global__ void rowwise_prod_by_num(	const double *dev_mat,
										const double *dev_vec,
										double *dev_result,
										int num_rows,
										int num_cols) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < num_rows * num_cols) {
		int row_index = index % num_rows;

		dev_result[index] = dev_mat[index] * dev_vec[row_index];

	}

}

__global__ void element_copy(	const double *dev_vec,
								double *dev_result,
								int length) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < length) {
		dev_result[index] = dev_vec[index];

	}
}

__global__ void element_scaling(	const double *dev_mat,
									double *dev_result,
									int length,
									double scaler) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < length) {
		dev_result[index] = dev_mat[index] * scaler;

	}
}

__global__ void element_shifting(	const double *dev_mat,
									double *dev_result,
									int length,
									double scaler) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < length) {
		dev_result[index] = dev_mat[index] + scaler;

	}
}

__global__ void element_squaring(	const double *dev_mat,
									double *dev_result,
									int length) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < length) {
		dev_result[index] = dev_mat[index] * dev_mat[index];

	}
}

__global__ void element_cubing(	const double *dev_mat,
								double *dev_result,
								int length) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < length) {
		dev_result[index] = dev_mat[index] * dev_mat[index] * dev_mat[index];

	}
}

__global__ void element_exponentiating(	const double *dev_mat,
										double *dev_result,
										int length) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < length) {
		double not_too_big_or_too_small =
				dev_mat[index] < _EXPONENT_CAP ? dev_mat[index] : _EXPONENT_CAP;

		not_too_big_or_too_small =
				not_too_big_or_too_small > _EXPONENT_BOT ? not_too_big_or_too_small : _EXPONENT_BOT;

		dev_result[index] = std::exp(not_too_big_or_too_small);

	}
}

__global__ void element_accu_prod(	const double *dev_mat,
									double *dev_result,
									int num_rows,
									int num_cols,
									double init_value) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < num_rows * num_cols) {
		int row_index = index % num_rows;
		int col_index = index / num_rows;

		dev_result[index] = init_value;
		for (int i = 0; i < row_index; i++)
			dev_result[index] *= dev_mat[IDX(i, col_index, num_rows)];

	}

}

__global__ void get_diag_element(	const double *dev_mat,
									double *dev_result,
									int num_rs_or_cs) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < num_rs_or_cs) {
		dev_result[index] = dev_mat[IDX(index, index, num_rs_or_cs)];

	}
}

__global__ void set_element_to_val(	double *dev_vec,
									int length,
									double val) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < length) {
		dev_vec[index] = val;

	}
}

// end of kernels

void elementwise_vec_prod(	const double *left_vec,
							const double *right_vec,
							double *dev_result,
							int length) {

	prod	<<<	(length + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
				_THREADS_PER_BLOCK
			>>>	(left_vec, right_vec, dev_result, length);

}

void elementwise_vec_sum(	const double *left_vec,
							const double *right_vec,
							double *dev_result,
							int length) {

	sum	<<<	(length + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
			_THREADS_PER_BLOCK
		>>>	(left_vec, right_vec, dev_result, length);

}

void elementwise_vec_sub(	const double *left_vec,
							const double *right_vec,
							double *dev_result,
							int length) {

	sub	<<<	(length + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
			_THREADS_PER_BLOCK
		>>>	(left_vec, right_vec, dev_result, length);

}

void elementwise_vec_div(	double const *left_vec,
							double const *right_vec,
							double *dev_result,
							int length) {

	div	<<<	(length + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
			_THREADS_PER_BLOCK
		>>>	(	left_vec,
				right_vec,
				dev_result,
				length);

}

void row_1st_pd(	const double *dev_mat,
					double *dev_result,
					int num_rows,
					int num_cols,
					double delta) {

	row_1st_diff	<<<	(num_rows * num_cols + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
						_THREADS_PER_BLOCK
					>>> (dev_mat, dev_result, num_rows, num_cols, delta);

}

void row_1st_inc(	double const *dev_mat,
					double *dev_result,
					int num_rows,
					int num_cols) {

	row_1st_ele_inc	<<<	(num_rows * num_cols + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
						_THREADS_PER_BLOCK
					>>> (dev_mat, dev_result, num_rows, num_cols);

}

void col_1st_pd(	const double *dev_mat,
					double *dev_result,
					int num_rows,
					int num_cols,
					double delta) {

	col_1st_diff	<<<	(num_rows * num_cols + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
						_THREADS_PER_BLOCK
					>>> (dev_mat, dev_result, num_rows, num_cols, delta);

}

void colwise_mat_shift_by_row_vec(	const double *dev_mat,
									const double *dev_vec,
									double *dev_result,
									int num_rows,
									int num_cols,
									double min_K,
									double delta) {

	set_vec_to_val(dev_result, num_rows * num_cols, 0.0);

	vertical_element_shift	<<<	(num_rows * num_cols + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
								_THREADS_PER_BLOCK
							>>> (dev_mat, dev_vec, dev_result, num_rows, num_cols, min_K, delta);

}

void colwise_sum_of_mat(	const double *dev_mat,
							double *dev_result,
							int num_rows,
							int num_cols) {

	colwise_sum	<<<	(num_cols + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
					_THREADS_PER_BLOCK
				>>>	(dev_mat, dev_result, num_rows, num_cols);

}

void colwise_mat_div_with_row_vec(	const double *dev_mat,
									const double *dev_vec,
									double *dev_result,
									double num_rows,
									double num_cols) {

	colwise_div_by_num	<<<	(num_rows * num_cols + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
							_THREADS_PER_BLOCK
						>>>	(dev_mat, dev_vec, dev_result, num_rows, num_cols);

}

void colwise_mat_prod_with_row_vec(	double const *dev_mat,
									double const *dev_vec,
									double *dev_result,
									double num_rows,
									double num_cols) {

	colwise_prod_by_num	<<<	(num_rows * num_cols + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
							_THREADS_PER_BLOCK
						>>>	(dev_mat, dev_vec, dev_result, num_rows, num_cols);

}

void rowwise_mat_prod_with_col_vec(	double const *dev_mat,
									double const *dev_vec,
									double *dev_result,
									double num_rows,
									double num_cols) {

	rowwise_prod_by_num	<<<	(num_rows * num_cols + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
							_THREADS_PER_BLOCK
						>>>	(dev_mat, dev_vec, dev_result, num_rows, num_cols);

}

void colwise_normalization(	double *dev_mat,
							double num_rows,
							double num_cols) {

	double *temp_dev_vec = NULL, *temp_dev_mat = NULL;

	dev_alloc(temp_dev_vec, "temp_dev_vec", num_cols);
	dev_alloc(temp_dev_mat, "temp_dev_mat", num_rows * num_cols);

	colwise_sum_of_mat(dev_mat, temp_dev_vec, num_rows, num_cols);
	colwise_mat_div_with_row_vec(	dev_mat,
									temp_dev_vec,
									temp_dev_mat,
									num_rows,
									num_cols);

	copy(temp_dev_mat, dev_mat, num_rows * num_cols);

	dev_release(temp_dev_vec, "temp_dev_vec");
	dev_release(temp_dev_mat, "temp_dev_mat");

}

void copy(	const double *dev_vec,
			double *dev_result,
			int length) {

	element_copy	<<<	(length + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
						_THREADS_PER_BLOCK
					>>>	(dev_vec, dev_result, length);

}

void vec_scaling( 	double const *dev_mat,
					double *dev_result,
					int length,
					double scaler) {

	element_scaling	<<<	(length + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
						_THREADS_PER_BLOCK
					>>>	(dev_mat, dev_result, length, scaler);

}

void vec_shifting( 	double const *dev_mat,
					double *dev_result,
					int length,
					double scaler)  {

	element_shifting	<<<	(length + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
							_THREADS_PER_BLOCK
						>>>	(dev_mat, dev_result, length, scaler);

}

void vec_squaring( 	double const *dev_mat,
					double *dev_result,
					int length) {

	element_squaring	<<<	(length + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
							_THREADS_PER_BLOCK
						>>>	(dev_mat, dev_result, length);

}

void vec_cubing( 	double const *dev_mat,
					double *dev_result,
					int length) {

	element_cubing	<<<	(length + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
						_THREADS_PER_BLOCK
					>>>	(dev_mat, dev_result, length);

}

void vec_exponentiating(	double const *dev_mat,
							double *dev_result,
							int length) {

	element_exponentiating	<<<	(length + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
								_THREADS_PER_BLOCK
							>>>	(dev_mat, dev_result, length);

}

void colwise_mat_accu_prod( const double *dev_mat,
							double *dev_result,
							int num_rows,
							int num_cols,
							double init_value) {

	element_accu_prod	<<<	(num_rows * num_cols + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
							_THREADS_PER_BLOCK
						>>>	(dev_mat, dev_result, num_rows, num_cols, init_value);

}

void get_diag_line_of_square_mat(	const double *dev_mat,
									double *dev_result,
									int num_rs_or_cs) {

	get_diag_element	<<<	(num_rs_or_cs + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
							_THREADS_PER_BLOCK
						>>>	(dev_mat, dev_result, num_rs_or_cs);

}

void vec_sum(	cublasHandle_t handle,
				const double *dev_vec,
				double *host_result,
				int length) {

	// start of declarations

	cublasStatus_t cublasStat;
	double *dev_one_vec = NULL;

	// end of declarations

	dev_alloc(dev_one_vec, "dev_one_vec", length);
	set_vec_to_val(dev_one_vec, length, 1.0);

	cublasStat = cublasDdot(handle, length, dev_vec, 1, dev_one_vec, 1, host_result);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS DOT failed in vec_sum.");
	}

	// release memory

	dev_release(dev_one_vec, "dev_one_vec");

}

void set_vec_to_val(	double *dev_vec,
						int length,
						double val) {

	set_element_to_val	<<<	(length + _THREADS_PER_BLOCK - 1) / _THREADS_PER_BLOCK,
							_THREADS_PER_BLOCK
						>>>	(dev_vec, length, val);

}

// allocate, deallocate and initialization methods

void dev_alloc_and_init(	const double * const &host_ptr,
							double *&dev_ptr,
							std::string name,
							int size) {

	cudaError_t cudaError;

	cudaError = cudaMalloc((void**)&dev_ptr, size * sizeof(double));
	if (cudaError != cudaSuccess) {
		std::cout << "Device memory allocation failed for " << name
				<< "." << std::endl;
	}

	cudaError = cudaMemcpy(	dev_ptr,
							host_ptr,
							size * sizeof(double),
							cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess) {
		std::cout << "Copy from host to device failed for " << name
				<< "." << std::endl;
	}

}

void dev_download(	double *&host_ptr,
					double * const &dev_ptr,
					std::string name,
					int size) {

	cudaError_t cudaError;

	cudaError = cudaMemcpy(	host_ptr,
							dev_ptr,
							size * sizeof(double),
							cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess) {
		std::cout << "Copy from device to host failed for " << name
				<< "." << std::endl;
	}

}

void dev_alloc(	double *&dev_ptr,
				std::string name,
				int size) {

	cudaError_t cudaError;

	cudaError = cudaMalloc((void**)&dev_ptr, size * sizeof(double));
	if (cudaError != cudaSuccess) {
		std::cout << "Device memory allocation failed for " << name
				<< "." << std::endl;
	}

}

void dev_release(	double *&dev_ptr,
					std::string name) {

	cudaError_t cudaError;

	if (dev_ptr != NULL) {

		cudaError = cudaFree(dev_ptr);

		if (cudaError != cudaSuccess) {
			std::cout << "Delete " << name
					<< " at " << dev_ptr
					<< ": "<< (cudaError ? "failed" : "succeeded")
					<< std::endl;
		}

	} else {
		std::cout << name << " is NULL." << std::endl;
	}
}

// tools for debugging

void dev_print(	double * const &dev_ptr,
				int num_rows,
				int num_cols) {

	// start of declarations

	cudaError_t cudaError;

	int i, j;
	double *host_ptr = new double[num_rows * num_cols];

	// end of declarations

	cudaError = cudaMemcpy(	host_ptr,
							dev_ptr,
							num_rows * num_cols * sizeof(double),
							cudaMemcpyDeviceToHost);

	if (cudaError != cudaSuccess) {
		std::cout << std::endl << dev_ptr << std::endl;
		std::cout << "Copy from device to host failed in dev_print." << std::endl;
	} else {
		std::cout << std::endl;
		for (i = 0; i < num_rows; i++) {
			for (j = 0; j < num_cols; j++) {
				printf("%+.4f ", host_ptr[IDX(i, j, num_rows)]);
			}
			printf("\n");
		}
	}

	delete[] host_ptr;

}

void print(	double * const &host_ptr,
			int num_rows,
			int num_cols) {

	std::cout << std::endl;
	for (int i = 0; i < num_rows; i++) {
		for (int j = 0; j < num_cols; j++) {
			printf("%+.4f ", host_ptr[IDX(i, j, num_rows)]);
		}
		printf("\n");
	}

}

void dev_print_3d(	double * const &dev_ptr,
					int num_rows,
					int num_cols,
					int num_pages) {

	// start of declarations

	cudaError_t cudaError;

	int i, j, k;
	double *host_ptr = new double[num_rows * num_cols * num_pages];

	// end of declarations

	cudaError = cudaMemcpy(	host_ptr,
							dev_ptr,
							num_rows * num_cols * num_pages * sizeof(double),
							cudaMemcpyDeviceToHost);

	if (cudaError != cudaSuccess) {
		std::cout << std::endl << dev_ptr << std::endl;
		std::cout << "Copy from device to host failed in dev_print." << std::endl;
	} else {
		std::cout << std::endl;
		for (i = 0; i < num_rows; i++) {
			for (j = 0; j < num_pages; j++) {
				for (k = 0; k < num_cols; k++) {
					printf("%+.4f ", host_ptr[IDXX(i, k, j, num_rows, num_cols)]);
				}
				std::cout << "\t\t";
			}
			std::cout << std::endl;
		}
	}

	delete[] host_ptr;

}

double dev_get(	double const *dev_ptr,
				int index) {

	// start of declarations

	cudaError_t cudaError;
	double host_ptr[1];

	// end of declarations

	cudaError = cudaMemcpy(	host_ptr,
							dev_ptr + index,
							sizeof(double),
							cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess) {
		std::cout << "Copy from device to host failed in dev_get." << std::endl;
	}

	return host_ptr[0];

}

