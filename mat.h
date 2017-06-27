//
// Created by xixuan on 4/6/17.
//

#ifndef MAT_H
#define MAT_H

#include <string>
#include <cublas_v2.h>

#define _THREADS_PER_BLOCK 256
#define _EXPONENT_CAP 10
#define _EXPONENT_BOT -10
#define _SMALL_NUM_BOT 1e-8

#define IDX(i, j, ld_i) (((j)*(ld_i))+(i))
#define IDXX(i, j, k, ld_i, ld_j) (((k)*((ld_i)*(ld_j)))+((j)*(ld_i))+(i))

////////////////////
// VECTOR-RELATED //
////////////////////

void elementwise_vec_prod(	double const *left_vec,
							double const *right_vec,
							double *dev_result,
							int length);
void elementwise_vec_sum(	double const *left_vec,
							double const *right_vec,
							double *dev_result,
							int length);
void elementwise_vec_sub(	double const *left_vec,
							double const *right_vec,
							double *dev_result,
							int length);
void elementwise_vec_div(	double const *left_vec,
							double const *right_vec,
							double *dev_result,
							int length);

void vec_scaling( 	double const *dev_mat,
					double *dev_result,
					int length,
					double scaler);

void vec_shifting( 	double const *dev_mat,
					double *dev_result,
					int length,
					double scaler);

void vec_squaring( 	double const *dev_mat,
					double *dev_result,
					int length);

void vec_cubing( 	double const *dev_mat,
					double *dev_result,
					int length);

void vec_exponentiating(	double const *dev_mat,
							double *dev_result,
							int length);

void vec_sum(	cublasHandle_t handle,
				const double *dev_vec,
				double *host_result,
				int length);

void copy(	double const *dev_vec,
			double *dev_result,
			int length);

void set_vec_to_val(	double *dev_vec,
						int length,
						double val);

////////////////////
// MATRIX-RELATED //
////////////////////

void row_1st_pd(	double const *dev_mat,
					double *dev_result,
					int num_rows,
					int num_cols,
					double delta);
void row_1st_inc(	double const *dev_mat,
					double *dev_result,
					int num_rows,
					int num_cols);

void col_1st_pd(	double const *dev_mat,
					double *dev_result,
					int num_rows,
					int num_cols,
					double delta);

void colwise_mat_shift_by_row_vec(	double const *dev_mat,
									double const *dev_vec,
									double *dev_result,
									int num_rows,
									int num_cols,
									double min_K,
									double delta);

void colwise_sum_of_mat(	double const *dev_mat,
							double *dev_result,
							int num_rows,
							int num_cols);

void colwise_mat_div_with_row_vec(	double const *dev_mat,
									double const *dev_vec,
									double *dev_result,
									double num_rows,
									double num_cols);

void colwise_mat_prod_with_row_vec(	double const *dev_mat,
									double const *dev_vec,
									double *dev_result,
									double num_rows,
									double num_cols);

void rowwise_mat_prod_with_col_vec(	double const *dev_mat,
									double const *dev_vec,
									double *dev_result,
									double num_rows,
									double num_cols);

void colwise_normalization(	double *dev_mat,
							double num_rows,
							double num_cols);

void colwise_mat_accu_prod( const double *dev_mat,
							double *dev_result,
							int num_rows,
							int num_cols,
							double init_value);

void get_diag_line_of_square_mat(	const double *dev_mat,
									double *dev_result,
									int num_rs_or_cs);

// allocate, deallocate and initialization methods

void dev_alloc_and_init(	double const * const &host_ptr,
							double *&dev_ptr,
							std::string name,
							int size);
void dev_alloc(	double *&dev_ptr,
				std::string name,
				int size);
void dev_release(	double *&dev_ptr,
					std::string name);
void dev_download(	double *&host_ptr,
					double * const &dev_ptr,
					std::string name,
					int size);

// tools for debugging

void print(	double * const &host_ptr,
			int num_rows,
			int num_cols);

void dev_print(	double * const &dev_ptr,
				int num_rows,
				int num_cols);
void dev_print_3d(	double * const &dev_ptr,
					int num_rows,
					int num_cols,
					int num_pages);

double dev_get(	double const *dev_ptr,
				int index);

#endif // MAT_H
