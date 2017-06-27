//
// Created by xixuan on 4/3/17.
//
// Note: Everything is column majored.
//

#include "model.h"
#include "mat.h"
#include "mul_nor_sampling.h"

#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cmath>

// initialization-related methods

void Sim::init_useful_matrices() {

	dev_alloc(	dev_one_mat_num_T_num_T,
				"dev_one_mat_num_T_num_T",
				num_T * num_T);
	set_vec_to_val(dev_one_mat_num_T_num_T, num_T * num_T, 1.0);

	dev_alloc(	dev_one_mat_num_K_num_T,
				"dev_one_mat_num_K_num_T",
				num_K * num_T);
	set_vec_to_val(dev_one_mat_num_K_num_T, num_K * num_T, 1.0);

	dev_alloc(	dev_zero_vec_num_K,
				"dev_zero_vec_num_K",
				num_K);
	set_vec_to_val(dev_zero_vec_num_K, num_K, 0.0);

}

void Sim::alloc_sim_vars() {

	dev_alloc(	dev_delta_w,
				"dev_delta_w",
				num_K * num_T);

	dev_alloc(	dev_delta_b,
				"dev_delta_b",
				num_T);

	dev_alloc(	dev_mu_f,
				"dev_mu_f",
				num_K * num_T);

	dev_alloc(	dev_all_mu_f,
				"dev_all_mu_f",
				num_K * num_T);

	dev_alloc(	dev_shifted_q_0,
				"dev_shifted_q_0",
				num_K * num_T);

	dev_alloc(	dev_1st_pd_of_shifted_q_0_wrt_K_over_q_0,
				"dev_1st_pd_of_shifted_q_0_wrt_K_over_q_0",
				num_K * num_T);

	dev_alloc(	dev_1st_pd_of_shifted_q_0_wrt_T_over_q_0,
				"dev_1st_pd_of_shifted_q_0_wrt_T_over_q_0",
				num_K * num_T);

	dev_alloc(	dev_2nd_pd_of_shifted_q_0_wrt_K_over_q_0,
				"dev_2nd_pd_of_shifted_q_0_wrt_K_over_q_0",
				num_K * num_T);

	dev_alloc(	dev_k,
				"dev_k",
				num_T);

	dev_alloc(	dev_1st_pd_of_k_wrt_T,
				"dev_1st_pd_of_k_wrt_T",
				num_T);

	dev_alloc(	dev_f,
				"dev_f",
				num_K * num_T);

	dev_alloc(	dev_1st_pd_of_f_wrt_T_over_f,
				"dev_1st_pd_of_f_wrt_T_over_f",
				num_K * num_T);

	dev_alloc(	dev_delta_p,
				"dev_delta_p",
				num_K * num_T);

	dev_alloc(	dev_p,
				"dev_p",
				num_K * num_T);

	nt_mat_w = new double[num_K * num_K];
	nt_mat_b = new double[1];

}

void Sim::alloc_temp_vars() {

	dev_alloc(	dev_A_temp_mat_of_num_K_num_K,
				"dev_A_temp_mat_of_num_K_num_K",
				num_K * num_K);
	dev_alloc(	dev_A_temp_mat_of_num_K_num_T,
				"dev_A_temp_mat_of_num_K_num_T",
				num_K * num_T);
	dev_alloc(	dev_A_temp_mat_of_num_T_num_T,
				"dev_A_temp_mat_of_num_T_num_T",
				num_T * num_T);
	dev_alloc(	dev_A_temp_vec_of_num_K,
				"dev_A_temp_vec_of_num_K",
				num_K);
	dev_alloc(	dev_A_temp_vec_of_num_T,
				"dev_A_temp_vec_of_num_T",
				num_T);

	dev_alloc(	dev_B_temp_mat_of_num_K_num_T,
				"dev_B_temp_mat_of_num_K_num_T",
				num_K * num_T);
	dev_alloc(	dev_B_temp_mat_of_num_T_num_T,
				"dev_B_temp_mat_of_num_T_num_T",
				num_T * num_T);
	dev_alloc(	dev_B_temp_vec_of_num_K,
				"dev_B_temp_vec_of_num_K",
				num_K);
	dev_alloc(	dev_B_temp_vec_of_num_T,
				"dev_B_temp_vec_of_num_T",
				num_T);

	dev_alloc(	dev_C_temp_mat_of_num_K_num_T,
				"dev_C_temp_mat_of_num_K_num_T",
				num_K * num_T);
	dev_alloc(	dev_C_temp_mat_of_num_T_num_T,
				"dev_C_temp_mat_of_num_T_num_T",
				num_T * num_T);
	dev_alloc(	dev_C_temp_vec_of_num_K,
				"dev_C_temp_vec_of_num_K",
				num_K);
	dev_alloc(	dev_C_temp_vec_of_num_T,
				"dev_C_temp_vec_of_num_T",
				num_T);

}

// public methods

Sim::Sim(	int num_K_val,
			double min_K_val,
			double max_K_val,
			int num_T_val,
			double min_T_val,
			double max_T_val,
			int device_index) {

	#if _DEBUG_MODEL
		std::cout 	<< "Sim model at " << this
						<< " initialized with"
						<< " num_K: " << num_K_val
						<< " min_K: " << min_K_val
						<< " max_K: " << max_K_val
						<< " delta_K: "
						<< ( (double) (max_K_val - min_K_val) ) / ( (double) (num_K_val - 1) )
						<< " num_T: " << num_T_val
						<< " min_T: " << min_T_val
						<< " max_T: " << max_T_val
						<< " delta_T: "
						<< ( (double) (max_T_val - min_T_val) ) / ( (double) (num_T_val - 1) )
						<< std::endl;
	#endif

	cublasStat = cublasCreate(&cublasHandle);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		std::cout << "CUBLAS initialization failed." << std::endl;
	}

	num_K = num_K_val;
	min_K = min_K_val;
	max_K = max_K_val;
	num_T = num_T_val;
	min_T = min_T_val;
	max_T = max_T_val;

	delta_K	= ( (double) (max_K - min_K) ) / ( (double) (num_K - 1) );
	delta_T = ( (double) (max_T - min_T) ) / ( (double) (num_T - 1) );

	dev_Ks = NULL;
	dev_Ts = NULL;
	dev_delta_b = NULL;
	dev_delta_w = NULL;

	dev_mu_k = NULL;
	dev_sigma_k = NULL;
	dev_sigma_f = NULL;
	dev_q_0 = NULL;
	dev_cor_w = NULL;

	host_cor_w = NULL;
	host_c = 0.0;

	// for simulation
	dev_mu_f = NULL;
	dev_all_mu_f = NULL;

	dev_shifted_q_0 = NULL;
	dev_1st_pd_of_shifted_q_0_wrt_K_over_q_0 = NULL;
	dev_1st_pd_of_shifted_q_0_wrt_T_over_q_0 = NULL;
	dev_2nd_pd_of_shifted_q_0_wrt_K_over_q_0 = NULL;

	dev_k = NULL;
	dev_1st_pd_of_k_wrt_T = NULL;

	dev_f = NULL;
	dev_1st_pd_of_f_wrt_T_over_f = NULL;

	dev_delta_p = NULL;
	dev_p = NULL;

	nt_mat_w = NULL;
	nt_mat_b = NULL;

	// useful matrices

	dev_one_mat_num_T_num_T = NULL;
	dev_one_mat_num_K_num_T = NULL;

	dev_zero_vec_num_K = NULL;

	// temporary storage
	dev_A_temp_mat_of_num_K_num_K = NULL;
	dev_A_temp_mat_of_num_K_num_T = NULL;
	dev_A_temp_mat_of_num_T_num_T = NULL;
	dev_A_temp_vec_of_num_K = NULL;
	dev_A_temp_vec_of_num_T = NULL;

	dev_B_temp_mat_of_num_K_num_T = NULL;
	dev_B_temp_mat_of_num_T_num_T = NULL;
	dev_B_temp_vec_of_num_K = NULL;
	dev_B_temp_vec_of_num_T = NULL;

	dev_C_temp_mat_of_num_K_num_T = NULL;
	dev_C_temp_mat_of_num_T_num_T = NULL;
	dev_C_temp_vec_of_num_K = NULL;
	dev_C_temp_vec_of_num_T = NULL;


	// start of declarations

	int i;
	double *host_values;

	// end of declarations

	// initialize dev_Ks

	host_values = new double[num_K];
	for (i = 0; i < num_K; i++) {
		host_values[i] = min_K + i * delta_K;
	}

	dev_alloc_and_init(	host_values,
						dev_Ks,
						"dev_Ks",
						num_K);

	delete[] host_values;

	// end of dev_Ks

	// initialize dev_Ts

	host_values = new double[num_T];
	for (i = 0; i < num_T; i++) {
		host_values[i] = min_T + i * delta_T;
	}

	dev_alloc_and_init(	host_values,
						dev_Ts,
						"dev_Ts",
						num_T);

	delete[] host_values;

	// end of dev_Ts

	// initialize useful matrices

	this->init_useful_matrices();

	// initialize matrices used for simulation

	this->alloc_sim_vars();
	this->alloc_temp_vars();

}

Sim::~Sim() {

	#if _DEBUG_MODEL
		std::cout 	<< "Sim model at "
					<< this
					<< " destroyed."
					<< std::endl;
	#endif

	// destroy cublasHandle
	cublasStat = cublasDestroy(cublasHandle);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		std::cout << "CUBLAS destroy failed." << std::endl;
	}

	// strikes and maturities

	dev_release(	dev_Ks,
					"dev_Ks");
	dev_release(	dev_Ts,
					"dev_Ts");

	// random factors

	dev_release(	dev_delta_b,
					"dev_delta_b");
	dev_release(	dev_delta_w,
					"dev_delta_w");

	// parameters

	dev_release(	dev_mu_k,
					"dev_mu_k");
	dev_release(	dev_sigma_k,
					"dev_sigma_k");
	dev_release(	dev_sigma_f,
					"dev_sigma_f");
	dev_release(	dev_q_0,
					"dev_q_0");
	dev_release(	dev_cor_w,
					"dev_cor_w");

	delete[] host_cor_w;
	// for simulation

	dev_release(	dev_mu_f,
					"dev_mu_f");
	dev_release(	dev_all_mu_f,
					"dev_all_mu_f");

	dev_release(	dev_shifted_q_0,
					"dev_shifted_q_0");
	dev_release(	dev_1st_pd_of_shifted_q_0_wrt_K_over_q_0,
					"dev_1st_pd_of_shifted_q_0_wrt_K_over_q_0");
	dev_release(	dev_1st_pd_of_shifted_q_0_wrt_T_over_q_0,
					"dev_1st_pd_of_shifted_q_0_wrt_T_over_q_0");
	dev_release(	dev_2nd_pd_of_shifted_q_0_wrt_K_over_q_0,
					"dev_2nd_pd_of_shifted_q_0_wrt_K_over_q_0");

	dev_release(	dev_k,
					"dev_k");
	dev_release(	dev_1st_pd_of_k_wrt_T,
					"dev_1st_pd_of_k_wrt_T");

	dev_release(	dev_f,
					"dev_f");
	dev_release(	dev_1st_pd_of_f_wrt_T_over_f,
					"dev_1st_pd_of_f_wrt_T_over_f");

	dev_release(	dev_delta_p,
					"dev_delta_p");
	dev_release(	dev_p,
					"dev_p");

	delete[] nt_mat_w;
	delete nt_mat_b;

	// useful matrices

	dev_release(	dev_one_mat_num_T_num_T,
					"dev_one_mat_num_T_num_T");
	dev_release(	dev_one_mat_num_K_num_T,
					"dev_one_mat_num_K_num_T");

	dev_release(	dev_zero_vec_num_K,
					"dev_zero_vec_num_K");

	// temporary storage

	dev_release(	dev_A_temp_mat_of_num_K_num_K,
					"dev_A_temp_mat_of_num_K_num_K");
	dev_release(	dev_A_temp_mat_of_num_K_num_T,
					"dev_A_temp_mat_of_num_K_num_T");
	dev_release(	dev_A_temp_mat_of_num_T_num_T,
					"dev_A_temp_mat_of_num_T_num_T");
	dev_release(	dev_A_temp_vec_of_num_K,
					"dev_A_temp_vec_of_num_K");
	dev_release(	dev_A_temp_vec_of_num_T,
					"dev_A_temp_vec_of_num_T");

	dev_release(	dev_B_temp_mat_of_num_K_num_T,
					"dev_B_temp_mat_of_num_K_num_T");
	dev_release(	dev_B_temp_mat_of_num_T_num_T,
					"dev_B_temp_mat_of_num_T_num_T");
	dev_release(	dev_B_temp_vec_of_num_K,
					"dev_B_temp_vec_of_num_K");
	dev_release(	dev_B_temp_vec_of_num_T,
					"dev_B_temp_vec_of_num_T");

	dev_release(	dev_C_temp_mat_of_num_K_num_T,
					"dev_C_temp_mat_of_num_K_num_T");
	dev_release(	dev_C_temp_mat_of_num_T_num_T,
					"dev_C_temp_mat_of_num_T_num_T");
	dev_release(	dev_C_temp_vec_of_num_K,
					"dev_C_temp_vec_of_num_K");
	dev_release(	dev_C_temp_vec_of_num_T,
					"dev_C_temp_vec_of_num_T");

}

void Sim::init(	const double *host_mu_k,
				const double *host_sigma_k,
				const double *host_sigma_f,
				const double *host_q_0,
				const double *host_input_cor_w,
				const double c) {
	/*
	 * Assumptions:
	 * 1.	mu_k is both t-varying and tau-varying
	 * 2.	sigma_k and sigma_f are tau-varying but not t-varying
	 * 3.	mu_f is both t-varying and tau-varying, decided when running
	 *
	 * So:
	 * 1.	mu_k should be a double array of the length num_T * num_T
	 * 		in the order of
	 * 		{{mu_k(t, tau_0)}, {mu_k(t, tau_1)}, ..., {mu_k(t, tau_num_T)}},
	 * 		where {mu_k(t, tau_i)} =
	 * 		{mu_k(t_0, tau_i), mu_k(t_1, tau_i), ..., mu_k(t_num_T, tau_i)}.
	 * 2.	sigma_k and sigma_f are simply double arrays of the length
	 * 		num_T like {sigma_f(tau_0), ..., sigma_f(tau_num_T)}.
	 * 3.	In the above setting, tau_0 > tau_1 > ... > tau_num_T.
	 */

	// dev_mu_k

	dev_alloc_and_init(	host_mu_k,
						dev_mu_k,
						"dev_mu_k",
						num_T * num_T);

	// dev_sigma_k

	dev_alloc_and_init(	host_sigma_k,
						dev_sigma_k,
						"dev_sigma_k",
						num_T);

	// dev_sigma_f

	dev_alloc_and_init(	host_sigma_f,
						dev_sigma_f,
						"dev_sigma_f",
						num_K * num_T);

	// dev_q_0

	dev_alloc_and_init(	host_q_0,
						dev_q_0,
						"dev_q_0",
						num_K * num_T);

	// check if host_input_cor_w is a valid correlation matrix

	#if _DEBUG_MODEL
		for (int j = 0; j < num_K; j++) {
			assert(host_input_cor_w[IDX(j, j, num_K)] == 1.0);
			for (int i = 0; i < j; i++) {
				assert(host_input_cor_w[IDX(i, j, num_K)] <= 1.0);
				assert(host_input_cor_w[IDX(i, j, num_K)] == host_input_cor_w[IDX(j, i, num_K)]);
			}
		}
	#endif

	// dev_cor_w

	dev_alloc_and_init(	host_input_cor_w,
						dev_cor_w,
						"dev_cor_w",
						num_K * num_K);

	// host_cor_w

	host_cor_w = new double[num_K * num_K];
	for (int i = 0; i < num_K * num_K; i++)
		host_cor_w[i] = host_input_cor_w[i];

	// host_c
	host_c = c;

	// nt_mat_w and nt_mat_b

	double *covar_array_for_w = new double[num_K * num_K];

	for (int j = 0; j < num_K; j++) {
		for (int i = 0; i < num_K; i++) {
			covar_array_for_w[IDX(i, j, num_K)] =
					host_cor_w[IDX(i, j, num_K)] * delta_T;
		}
	}

	norm_transform_mat(	nt_mat_w,
						covar_array_for_w,
						num_K);

	nt_mat_b[0] = std::sqrt(delta_T);

	delete[] covar_array_for_w;

}

void Sim::simulate(Sim ** const &models, int model_index, int random_seed) {

	this->deltas(random_seed);

	this->sim_k();

	this->init_mu_f();

	for (int i = 0; i < num_T; i++) {

		this->sim_f(models, model_index, i);
		this->sim_p(i);
		this->sim_mu_f(i);

	}

	// \Delta p
	row_1st_inc(	dev_p,
					dev_delta_p,
					num_K, num_T);

//	dev_print(dev_p, num_K, num_T);

}

void Sim::deltas(int random_seed, bool show_summary) {

	// start of declarations

	double *result_array_for_w = new double[num_K * num_T];

	double *result_array_for_b = new double[num_T];

	// end of declarations

	// sample for dev_delta_w

	draw_from_mul_gaussian(	result_array_for_w,
							num_K,
							num_T,
							nt_mat_w,
							random_seed,
							show_summary);

	cudaError = cudaMemcpy(	dev_delta_w,
							result_array_for_w,
							num_K * num_T * sizeof(double),
							cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess) {
		std::cout << "Copy from host to device failed for dev_delta_w." << std::endl;
		this->~Sim();
	}

	// end of sampling for dev_delta_w

	// sample for dev_delta_w

	draw_from_mul_gaussian(	result_array_for_b,
							1,
							num_T,
							nt_mat_b,
							random_seed,
							show_summary);

	cudaError = cudaMemcpy(	dev_delta_b,
							result_array_for_b,
							num_T * sizeof(double),
							cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess) {
		std::cout << "Copy from host to device failed for dev_delta_w." << std::endl;
		this->~Sim();
	}

	// end of sampling for dev_delta_b

	delete[] result_array_for_w;

	delete[] result_array_for_b;

}

void Sim::sim_k() {

	// \sigma^2
	vec_squaring(	dev_sigma_k,
					dev_A_temp_vec_of_num_T,
					num_T);

	// \frac{1}{2}\sigma^2
	vec_scaling(	dev_A_temp_vec_of_num_T,
					dev_B_temp_vec_of_num_T,
					num_T,
					0.5);

	// \frac{1}{2}\sigma^2 - \mu
	colwise_mat_prod_with_row_vec(	dev_one_mat_num_T_num_T,
									dev_B_temp_vec_of_num_T,
									dev_A_temp_mat_of_num_T_num_T,
									num_T,
									num_T);
	elementwise_vec_sub(	dev_A_temp_mat_of_num_T_num_T,
							dev_mu_k,
							dev_B_temp_mat_of_num_T_num_T,
							num_T * num_T);

	// ( \frac{1}{2}\sigma^2 - \mu ) \delta t
	vec_scaling(	dev_B_temp_mat_of_num_T_num_T,
					dev_A_temp_mat_of_num_T_num_T,
					num_T * num_T,
					delta_T);

	// \sigma * \delta B
	rowwise_mat_prod_with_col_vec(	dev_one_mat_num_T_num_T,
									dev_delta_b,
									dev_B_temp_mat_of_num_T_num_T,
									num_T,
									num_T);

	colwise_mat_prod_with_row_vec(	dev_B_temp_mat_of_num_T_num_T,
									dev_sigma_k,
									dev_C_temp_mat_of_num_T_num_T,
									num_T,
									num_T);

	// ( \frac{1}{2}\sigma^2 - \mu ) \delta t - \sigma * \delta B
	elementwise_vec_sub(	dev_A_temp_mat_of_num_T_num_T,
							dev_C_temp_mat_of_num_T_num_T,
							dev_B_temp_mat_of_num_T_num_T,
							num_T * num_T);

	// exp( ( \frac{1}{2}\sigma^2 - \mu ) \delta t - \sigma * \delta B )
	vec_exponentiating(	dev_B_temp_mat_of_num_T_num_T,
						dev_A_temp_mat_of_num_T_num_T,
						num_T * num_T);

	// accumulate
	colwise_mat_accu_prod(	dev_A_temp_mat_of_num_T_num_T,
							dev_B_temp_mat_of_num_T_num_T,
							num_T,
							num_T,
							1.0);
	get_diag_line_of_square_mat(	dev_B_temp_mat_of_num_T_num_T,
									dev_k,
									num_T);

	// dev_1st_pd_of_k_wrt_T
	row_1st_pd(	dev_B_temp_mat_of_num_T_num_T,
				dev_A_temp_mat_of_num_T_num_T,
				num_T,
				num_T,
				delta_T);
	get_diag_line_of_square_mat(	dev_A_temp_mat_of_num_T_num_T,
									dev_A_temp_vec_of_num_T,
									num_T);

	//////////////////////////////////////
	// MAY NEED A MINUS SIGN LIKE THIS! //
	//////////////////////////////////////
	vec_scaling(	dev_A_temp_vec_of_num_T,
					dev_1st_pd_of_k_wrt_T,
					num_T,
					-1.0);

	// dev_shifted_q_0
	colwise_mat_shift_by_row_vec(	dev_q_0,
									dev_k,
									dev_A_temp_mat_of_num_K_num_T,
									num_K,
									num_T,
									min_K,
									delta_K);

	colwise_normalization(	dev_A_temp_mat_of_num_K_num_T,
							num_K,
							num_T);

	vec_scaling(	dev_A_temp_mat_of_num_K_num_T,
					dev_shifted_q_0,
					num_K * num_T,
					1.0 / delta_K);

	// dev_1st_pd_of_shifted_q_0_wrt_T_over_q_0
	row_1st_pd(	dev_shifted_q_0,
				dev_A_temp_mat_of_num_K_num_T,
				num_K,
				num_T,
				delta_T);

	elementwise_vec_div(	dev_A_temp_mat_of_num_K_num_T,
							dev_shifted_q_0,
							dev_B_temp_mat_of_num_K_num_T,
							num_K * num_T);
	//////////////////////////////////////
	// MAY NEED A MINUS SIGN LIKE THIS! //
	//////////////////////////////////////
	vec_scaling(	dev_B_temp_mat_of_num_K_num_T,
					dev_1st_pd_of_shifted_q_0_wrt_T_over_q_0,
					num_K * num_T,
					-1.0);

	// dev_1st_pd_of_shifted_q_0_wrt_K_over_q_0
	col_1st_pd(	dev_shifted_q_0,
				dev_A_temp_mat_of_num_K_num_T,
				num_K,
				num_T,
				delta_K);
	elementwise_vec_div(	dev_A_temp_mat_of_num_K_num_T,
							dev_shifted_q_0,
							dev_1st_pd_of_shifted_q_0_wrt_K_over_q_0,
							num_K * num_T);

	// dev_2nd_pd_of_shifted_q_0_wrt_K_over_q_0

	col_1st_pd(	dev_A_temp_mat_of_num_K_num_T,
				dev_B_temp_mat_of_num_K_num_T,
				num_K,
				num_T,
				delta_K);

	elementwise_vec_div(	dev_B_temp_mat_of_num_K_num_T,
							dev_shifted_q_0,
							dev_2nd_pd_of_shifted_q_0_wrt_K_over_q_0,
							num_K * num_T);

}

void Sim::init_mu_f() {

	// (2)+(3)+(4)
	// K(ln q)_3^\prime(0,T−t,Kk(t,T−t))( (\sigma_k^2(t,T−t)-\mu_k(t,T−t))k(t,T−t) - k_2^\prime(t,T-t) )

	// \sigma_k^2(t, T − t)
	vec_squaring(	dev_sigma_k,
					dev_A_temp_vec_of_num_T,
					num_T);
	// \sigma_k^2(t, T − t)-\mu_k(t, T − t)
	get_diag_line_of_square_mat(	dev_mu_k,
									dev_C_temp_vec_of_num_T,
									num_T);
	elementwise_vec_sub(	dev_A_temp_vec_of_num_T,
							dev_C_temp_vec_of_num_T,
							dev_B_temp_vec_of_num_T,
							num_T);
	// (\sigma_k^2(t, T − t)-\mu_k(t, T − t))k(t, T − t)
	elementwise_vec_prod(	dev_B_temp_vec_of_num_T,
							dev_k,
							dev_A_temp_vec_of_num_T,
							num_T);
	// (\sigma_k^2(t,T−t)-\mu_k(t,T−t))k(t,T−t) - k_2^\prime(t,T-t)
	elementwise_vec_sub(	dev_A_temp_vec_of_num_T,
							dev_1st_pd_of_k_wrt_T,
							dev_B_temp_vec_of_num_T,
							num_T);

	// (ln q)_3^\prime(0,T−t,Kk(t,T−t))( (\sigma_k^2(t,T−t)-\mu_k(t,T−t))k(t,T−t) - k_2^\prime(t,T-t) )
	colwise_mat_prod_with_row_vec(	dev_1st_pd_of_shifted_q_0_wrt_K_over_q_0,
									dev_B_temp_vec_of_num_T,
									dev_A_temp_mat_of_num_K_num_T,
									num_K,
									num_T);
	// K(ln q)_3^\prime(0, T − t, Kk(t, T − t))(\sigma_k^2(t, T − t)-\mu_k(t, T − t))k(t, T − t)
	rowwise_mat_prod_with_col_vec(	dev_A_temp_mat_of_num_K_num_T,
									dev_Ks,
									dev_B_temp_mat_of_num_K_num_T,			// (2)+(3)+(4)
									num_K,
									num_T);

	// (1) −(ln q)_2^\prime(0, T − t, Kk(t, T − t))
	elementwise_vec_sub(	dev_B_temp_mat_of_num_K_num_T,
							dev_1st_pd_of_shifted_q_0_wrt_T_over_q_0,
							dev_A_temp_mat_of_num_K_num_T,					// (1)+(2)+(3)+(4)
							num_K * num_T);

	// (5) 1/2 K^2 q_{33}^{\prime\prime}/q_0 (0, T − t, Kk(t, T − t)) \sigma_k^2 (t, T − t) k^2(t, T − t)

	// \sigma_k (t, T − t) k(t, T − t)
	elementwise_vec_prod(	dev_sigma_k,
							dev_k,
							dev_A_temp_vec_of_num_T,
							num_T);
	// \sigma_k^2 (t, T − t) k^2(t, T − t)
	vec_squaring(	dev_A_temp_vec_of_num_T,
					dev_B_temp_vec_of_num_T,
					num_T);

	// q_{33}^{\prime\prime}/q_0 (0, T − t, Kk(t, T − t)) \sigma_k^2 (t, T − t) k^2(t, T − t)
	colwise_mat_prod_with_row_vec(	dev_2nd_pd_of_shifted_q_0_wrt_K_over_q_0,
									dev_B_temp_vec_of_num_T,
									dev_B_temp_mat_of_num_K_num_T,
									num_K,
									num_T);

	// K^2
	vec_squaring(	dev_Ks,
					dev_A_temp_vec_of_num_K,
					num_K);
	// 1/2 K^2
	vec_scaling(	dev_A_temp_vec_of_num_K,
					dev_B_temp_vec_of_num_K,
					num_K,
					0.5);
	// 1/2 K^2 q_{33}^{\prime\prime}/q_0 (0, T − t, Kk(t, T − t)) \sigma_k^2 (t, T − t) k^2(t, T − t)
	rowwise_mat_prod_with_col_vec(	dev_B_temp_mat_of_num_K_num_T,
									dev_B_temp_vec_of_num_K,
									dev_C_temp_mat_of_num_K_num_T,			// (5)
									num_K,
									num_T);

	// (1)+(2)+(3)+(4)+(5)
	elementwise_vec_sum(	dev_A_temp_mat_of_num_K_num_T,
							dev_C_temp_mat_of_num_K_num_T,
							dev_B_temp_mat_of_num_K_num_T,
							num_K * num_T);

	vec_shifting(	dev_B_temp_mat_of_num_K_num_T,
					dev_mu_f,
					num_K * num_T,
					host_c);

}

void Sim::sim_f(Sim ** const &models, int model_index, int col_index) {

	if (col_index == 0) {

		set_vec_to_val(dev_f, num_K * num_T, 1.0);
		set_vec_to_val(dev_1st_pd_of_f_wrt_T_over_f, num_K * num_T, 0.0);

	}
	else {

		// start of declarations

		int i;

		double *dev_j_m_1_th_col_of_mu_f =
				dev_mu_f + IDX(0, col_index - 1, num_K);
		double *dev_j_m_1_th_col_of_delta_w =
				dev_delta_w + IDX(0, col_index - 1, num_K);

		double *dev_j_th_col_of_1st_pd_of_f_wrt_T_over_f =
				dev_1st_pd_of_f_wrt_T_over_f + IDX(0, col_index, num_K);

		// end of declarations

		// start of making dev_all_mu_f

		for(i = 0; i < col_index; i++)
			copy(	dev_j_m_1_th_col_of_mu_f,
					dev_all_mu_f + IDX(0, i, num_K),
					num_K);

		for(i = col_index; i < num_T; i++) {
			copy(	models[model_index - (i - col_index + 1)]
			     	       ->get_dev_ptr_to_col_of_mu_f(col_index - 1),
					dev_all_mu_f + IDX(0, i, num_K),
					num_K);
		}

		// end of making dev_all_mu_f

		// \sigma_f^2
		vec_squaring(	dev_sigma_f,
						dev_A_temp_mat_of_num_K_num_T,
						num_K * num_T);
		// 1/2 \sigma_f^2
		vec_scaling(	dev_A_temp_mat_of_num_K_num_T,
						dev_B_temp_mat_of_num_K_num_T,
						num_K * num_T,
						0.5);
		// \mu_f - 1/2 \sigma_f^2
		elementwise_vec_sub(	dev_all_mu_f,
								dev_B_temp_mat_of_num_K_num_T,
								dev_C_temp_mat_of_num_K_num_T,
								num_K * num_T);
		// (\mu_f - 1/2 \sigma_f^2) \Delta t
		vec_scaling(	dev_C_temp_mat_of_num_K_num_T,
						dev_A_temp_mat_of_num_K_num_T,
						num_K * num_T,
						delta_T);

		// \sigma_f * \Delta w
		rowwise_mat_prod_with_col_vec(	dev_sigma_f,
										dev_j_m_1_th_col_of_delta_w,
										dev_C_temp_mat_of_num_K_num_T,
										num_K,
										num_T);

		// (\mu_f - 1/2 \sigma_f^2) \Delta t + \sigma_f * \Delta w
		elementwise_vec_sum(	dev_A_temp_mat_of_num_K_num_T,
								dev_C_temp_mat_of_num_K_num_T,
								dev_B_temp_mat_of_num_K_num_T,
								num_K * num_T);

		// exp((\mu_f - 1/2 \sigma_f^2) \Delta t + \sigma_f * \Delta w)
		vec_exponentiating(	dev_B_temp_mat_of_num_K_num_T,
							dev_A_temp_mat_of_num_K_num_T,
							num_K * num_T);

		// f * exp((\mu_f - 1/2 \sigma_f^2) \Delta t + \sigma_f * \Delta w)
		elementwise_vec_prod(	dev_f,
								dev_A_temp_mat_of_num_K_num_T,
								dev_B_temp_mat_of_num_K_num_T,
								num_K * num_T);

		// copy to f
		copy(	dev_B_temp_mat_of_num_K_num_T,
				dev_f, num_K * num_T);

		// dev_1st_pd_of_f_wrt_T_over_f
		row_1st_pd(	dev_f,
					dev_A_temp_mat_of_num_K_num_T,
					num_K,
					num_T,
					delta_T);
		elementwise_vec_div(	dev_A_temp_mat_of_num_K_num_T,
								dev_f,
								dev_B_temp_mat_of_num_K_num_T,
								num_K * num_T);

		//////////////////////////////////////
		// MAY NEED A MINUS SIGN LIKE THIS! //
		//////////////////////////////////////
		vec_scaling(	dev_B_temp_mat_of_num_K_num_T + IDX(0, col_index, num_K),
						dev_j_th_col_of_1st_pd_of_f_wrt_T_over_f,
						num_K,
						-1.0);

	}
}

void Sim::sim_p(int col_index) {

	if (col_index == 0)
		copy(dev_shifted_q_0, dev_p, num_K);
	else {

		// start of declarations

		double *dev_j_th_col_of_shifted_q_0 = dev_shifted_q_0 + IDX(0, col_index, num_K);
		double *dev_j_th_col_of_f = dev_f + IDX(0, col_index, num_K);
		double *dev_j_th_col_of_p = dev_p + IDX(0, col_index, num_K);

		// end of declarations

		// q(0, τ, Kk(t, τ ))f (t, τ, K)
		elementwise_vec_prod(	dev_j_th_col_of_shifted_q_0,
								dev_j_th_col_of_f,
								dev_A_temp_vec_of_num_K,
								num_K);

		// ( q(0, τ, Kk(t, τ ))f (t, τ, K) ) / a(t, τ)
		colwise_normalization(	dev_A_temp_vec_of_num_K,
								num_K,
								1);
		vec_scaling(	dev_A_temp_vec_of_num_K,
						dev_j_th_col_of_p,
						num_K,
						1.0 / delta_K);

//		dev_print(dev_j_th_col_of_p, 1, num_K);

//		// martingale property
//		double *dev_j_m_1_th_col_of_delta_w = dev_delta_w + IDX(0, col_index - 1, num_K);
//		double *dev_j_m_1_th_col_of_p = dev_p + IDX(0, col_index - 1, num_K);
//		double *dev_j_m_1_th_col_of_sigma_f = dev_sigma_f + IDX(0, col_index - 1, num_K);
//
//		double *dev_j_m_1_th_col_of_1st_pd_of_shifted_q_0_wrt_K_over_q_0 =
//				dev_1st_pd_of_shifted_q_0_wrt_K_over_q_0 + IDX(0, col_index - 1, num_K);
//
//		double j_m_1_th_of_sigma_k = dev_get(dev_sigma_k, col_index - 1);
//		double j_m_1_th_of_k = dev_get(dev_k, col_index - 1);
//		double j_m_1_th_of_delta_b = dev_get(dev_delta_b, col_index - 1);
//
//		double scaler_0;
//		double scaler_1;
//
//		// σ f (t, T − t, K)dW (t, K)
//		elementwise_vec_prod(	dev_j_m_1_th_col_of_sigma_f,
//								dev_j_m_1_th_col_of_delta_w,
//								dev_A_temp_vec_of_num_K,
//								num_K);
//
//		// −K(ln q) 0 3 (0, T − t, Kk(t, T − t))σ k (t, T − t)k(t, T − t)dB(t)
//		elementwise_vec_prod(	dev_Ks,
//								dev_j_m_1_th_col_of_1st_pd_of_shifted_q_0_wrt_K_over_q_0,
//								dev_B_temp_vec_of_num_K,
//								num_K);
//		vec_scaling(	dev_B_temp_vec_of_num_K,
//						dev_C_temp_vec_of_num_K,
//						num_K,
//						j_m_1_th_of_sigma_k * j_m_1_th_of_k * j_m_1_th_of_delta_b);
//		elementwise_vec_sub(	dev_A_temp_vec_of_num_K,
//								dev_C_temp_vec_of_num_K,
//								dev_B_temp_vec_of_num_K,
//								num_K);
//
//		// * p(t, T − t, K)
//		elementwise_vec_prod(	dev_B_temp_vec_of_num_K,
//								dev_j_m_1_th_col_of_p,
//								dev_C_temp_vec_of_num_K,					// (1) + (2)
//								num_K);
//
//		// \int L(ln q) 0 3 (0, T − t, Lk(t, T − t))p(t, T − t, L)dL dB(t)
//		elementwise_vec_prod(	dev_Ks,
//								dev_j_m_1_th_col_of_1st_pd_of_shifted_q_0_wrt_K_over_q_0,
//								dev_A_temp_vec_of_num_K,
//								num_K);
//		elementwise_vec_prod(	dev_A_temp_vec_of_num_K,
//								dev_j_m_1_th_col_of_p,
//								dev_B_temp_vec_of_num_K,
//								num_K);
//		vec_sum(	cublasHandle,
//					dev_B_temp_vec_of_num_K,
//					&scaler_0,
//					num_K);
//
//		// \int p(t, T − t, L)σ f (t, T − t, L)dW (t, L) dL.
//		elementwise_vec_prod(	dev_j_m_1_th_col_of_p,
//								dev_j_m_1_th_col_of_sigma_f,
//								dev_A_temp_vec_of_num_K,
//								num_K);
//		elementwise_vec_prod(	dev_A_temp_vec_of_num_K,
//								dev_j_m_1_th_col_of_delta_w,
//								dev_B_temp_vec_of_num_K,
//								num_K);
//		vec_sum(	cublasHandle,
//					dev_B_temp_vec_of_num_K,
//					&scaler_1,
//					num_K);
//
//		// * p
//
//		vec_scaling(	dev_j_m_1_th_col_of_p,
//						dev_B_temp_vec_of_num_K,
//						num_K,
//						j_m_1_th_of_sigma_k * j_m_1_th_of_k * j_m_1_th_of_delta_b * scaler_0 - scaler_1);
//
//		// (1) + (2) + (3) + (4)
//		elementwise_vec_sum(	dev_C_temp_vec_of_num_K,
//								dev_B_temp_vec_of_num_K,
//								dev_A_temp_vec_of_num_K,
//								num_K);
//
//		dev_print(dev_A_temp_vec_of_num_K, 1, num_K);

	}

}

void Sim::sim_mu_f(int col_index) {

	// start of declarations

	double *dev_j_th_col_of_1st_pd_of_f_wrt_T_over_f =
			dev_1st_pd_of_f_wrt_T_over_f + IDX(0, col_index, num_K);
	double *dev_j_th_col_of_1st_pd_of_shifted_q_0_wrt_K_over_q_0 =
			dev_1st_pd_of_shifted_q_0_wrt_K_over_q_0 + IDX(0, col_index, num_K);

	double *dev_j_th_col_of_p = dev_p + IDX(0, col_index, num_K);
	double *dev_j_th_col_of_sigma_f = dev_sigma_f + IDX(0, col_index, num_K);

	double *dev_j_th_col_of_mu_f = dev_mu_f + IDX(0, col_index, num_K);

	double scaler;
	double j_th_of_sigma_k = dev_get(dev_sigma_k, col_index);
	double j_th_of_k = dev_get(dev_k, col_index);

	// end of declarations

	// (2)
	// (ln q)_3^\prime(0, T − t, Lk(t, T − t))p(t, T − t, L)
	elementwise_vec_prod(	dev_j_th_col_of_1st_pd_of_shifted_q_0_wrt_K_over_q_0,
							dev_j_th_col_of_p,
							dev_A_temp_vec_of_num_K,
							num_K);
	// L(ln q)_^\prime (0, T − t, Lk(t, T − t))p(t, T − t, L)
	elementwise_vec_prod(	dev_A_temp_vec_of_num_K,
							dev_Ks,
							dev_B_temp_vec_of_num_K,
							num_K);
	// \int_{\mathbb{R}_{\geq 0}}L(ln q) 0 3 (0, T − t, Lk(t, T − t))p(t, T − t, L)dL
	vec_sum(	cublasHandle,
				dev_B_temp_vec_of_num_K,
				&scaler,
				num_K);
	// 	σ_k^2(t, T − t)k^2(t, T − t)\int_{\mathbb{R}_{\geq 0}}L(ln q) 0 3 (0, T − t, Lk(t, T − t))p(t, T − t, L)dL
	scaler = 	scaler *
				delta_K *
				j_th_of_sigma_k *
				j_th_of_sigma_k *
				j_th_of_k *
				j_th_of_k *
				-1.0;						// changed its sign here!!

	// K(ln q)_3^\prime(0, T − t, Kk(t, T − t))
	elementwise_vec_prod(	dev_j_th_col_of_1st_pd_of_shifted_q_0_wrt_K_over_q_0,
							dev_Ks,
							dev_A_temp_vec_of_num_K,
							num_K);
	// (2)
	vec_scaling(	dev_A_temp_vec_of_num_K,
					dev_B_temp_vec_of_num_K,											// (2)
					num_K,
					scaler);

	// (1)+(2)
	elementwise_vec_sub(	dev_B_temp_vec_of_num_K,
							dev_j_th_col_of_1st_pd_of_f_wrt_T_over_f,
							dev_A_temp_vec_of_num_K,									// (1)+(2)
							num_K);

	// (3)
	// p(t, T − t, L)σ_f(t, T − t, L)
	elementwise_vec_prod(	dev_j_th_col_of_p,
							dev_j_th_col_of_sigma_f,
							dev_B_temp_vec_of_num_K,
							num_K);
	// p(t, T − t, L)σ f (t, T − t, L)c_W(t, L, K)
	rowwise_mat_prod_with_col_vec(	dev_cor_w,
									dev_B_temp_vec_of_num_K,
									dev_A_temp_mat_of_num_K_num_K,
									num_K,
									num_K);
	// \int_{\mathbb{R}_{\geq 0}} p(t, T − t, L)σ f (t, T − t, L)c_W(t, L, K) dL
	colwise_sum_of_mat(	dev_A_temp_mat_of_num_K_num_K,
						dev_B_temp_vec_of_num_K,
						num_K,
						num_K);
	vec_scaling(dev_B_temp_vec_of_num_K,
				dev_A_temp_vec_of_num_K,
				num_K,
				delta_K);
	// σ_f(t, T − t, K)\int_{\mathbb{R}_{\geq 0}} p(t, T − t, L)σ f (t, T − t, L)c_W(t, L, K) dL
	elementwise_vec_prod(	dev_j_th_col_of_sigma_f,
							dev_A_temp_vec_of_num_K,
							dev_C_temp_vec_of_num_K,									// (3)
							num_K);

	// (1)+(2)+(3)
	elementwise_vec_sub(	dev_A_temp_vec_of_num_K,
							dev_C_temp_vec_of_num_K,
							dev_B_temp_vec_of_num_K,									// (1)+(2)+(3)
							num_K);

	// initialized (1)+(2)+(3)+(4)+(5) + updated (1)+(2)+(3)
	elementwise_vec_sum(	dev_j_th_col_of_mu_f,
							dev_B_temp_vec_of_num_K,
							dev_j_th_col_of_mu_f,
							num_K);

}

double const *Sim::get_dev_ptr_to_col_of_p(const int col_index) const {
#if _DEBUG_MODEL
	assert(col_index >= 0 && col_index < num_T);
#endif
	return dev_p + IDX(0, col_index, num_K);
}

double const *Sim::get_dev_ptr_to_col_of_delta_p(const int col_index) const {
#if _DEBUG_MODEL
	assert(col_index >= 0 && col_index < num_T);
#endif
	return dev_delta_p + IDX(0, col_index, num_K);
}

double const *Sim::get_dev_ptr_to_col_of_mu_f(const int col_index) const {
#if _DEBUG_MODEL
	assert(col_index >= 0 && col_index < num_T);
#endif
	return dev_mu_f + IDX(0, col_index, num_K);
}



