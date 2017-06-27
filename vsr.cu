//
// Created by xixuan on 4/3/17.
//
// Note: everything is column majored.
//

#include "vsr.h"
#include "mat.h"
#include "tools.h"

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cassert>
#include <cmath>
#include <string>

// initialization-related methods

void Vsr::init_useful_matrices() {

	// start of declarations

	int i, j;
	double *host_vec;
	double alpha, beta;

	// end of declarations

	// dev_one_mat_num_T_num_T and dev_one_mat_num_K_num_T

	dev_alloc(	dev_one_mat_num_T_num_T,
				"dev_one_mat_num_T_num_T",
				num_T * num_T);

	set_vec_to_val(	dev_one_mat_num_T_num_T,
					num_T * num_T,
					1.0);

	dev_alloc(	dev_one_mat_num_K_num_T,
				"dev_one_mat_num_K_num_T",
				num_K * num_T);

	set_vec_to_val(	dev_one_mat_num_K_num_T,
					num_K * num_T,
					1.0);

	// end of dev_one_mat_num_T_num_T and dev_one_mat_num_K_num_T

	// dev_theta_1 and dev_theta_1_squared

	host_vec = new double[num_K * num_K];

	for (j = 0; j < num_K; j++)
		for (i = 0; i < num_K; i++) {
			if (i <= j)
				host_vec[IDX(i, j, num_K)] = delta_K;
			else
				host_vec[IDX(i, j, num_K)] = 0.0;
		}
	dev_alloc_and_init(	host_vec,
						dev_theta_1,
						"dev_theta_1",
						num_K * num_K);

	delete[] host_vec;

	dev_alloc(	dev_theta_1_squared,
				"dev_theta_1_squared",
				num_K * num_K);

	alpha = 1.0;
	beta = 0.0;
	cublasStat = cublasDgemm(	cublasHandle,
								CUBLAS_OP_N, CUBLAS_OP_N,
								num_K, num_K, num_K,
								&alpha,
								dev_theta_1, num_K,
								dev_theta_1, num_K,
								&beta,
								dev_theta_1_squared, num_K);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("cublasDgemm in init_useful_matrices failed.");
	}

	// end of dev_theta_1 and dev_theta_1_squared

	// dev_mu_delta_t and dev_lambda

	dev_alloc(	dev_mu_delta_t,
				"dev_mu_delta_t",
				roi_num_T);
	set_vec_to_val(	dev_mu_delta_t,
					roi_num_T,
					delta_T);

	dev_alloc(	dev_lambda,
				"dev_lambda",
				num_K);
	set_vec_to_val(	dev_lambda,
					num_K,
					1.0);
	elementwise_vec_div(	dev_lambda,
							dev_Ks,
							dev_A_temp_vec_of_num_K,
							num_K);
	vec_squaring(	dev_A_temp_vec_of_num_K,
					dev_B_temp_vec_of_num_K,
					num_K);
	vec_scaling(	dev_B_temp_vec_of_num_K,
					dev_lambda,
					num_K,
					delta_K);

	// end of dev_mu_delta_t and dev_lambda

}

void Vsr::alloc_sim_vars() {

	// start of declarations

	int i;

	// end of declarations

	models = new Sim*[num_T];
	for (i = 0; i < num_T; i++) {
		models[i] = new Sim(	num_K,
								min_K,
								max_K,
								1 + i,
								min_T,
								i * delta_T + min_T,
								chosen_device_index);
	}

	dev_alloc(	dev_vec_delta_p,
				"dev_vec_delta_p",
				num_K * roi_num_T * num_pages);
	dev_alloc(	dev_vec_p,
				"dev_vec_p",
				num_K * roi_num_T * num_pages);

	dev_alloc(	dev_vec_1st_pd_of_p_wrt_T,
				"dev_vec_1st_pd_of_p_wrt_T",
				num_K * roi_num_T * num_pages);
	dev_alloc(	dev_vec_2nd_pd_of_p_wrt_T,
				"dev_vec_2nd_pd_of_p_wrt_T",
				num_K * roi_num_T * num_pages);
	dev_alloc(	dev_vec_3rd_pd_of_p_wrt_T,
				"dev_vec_3rd_pd_of_p_wrt_T",
				num_K * roi_num_T * num_pages);

	host_delta_vsr = new double[num_pages - 1];

}

void Vsr::alloc_temp_vars() {

	dev_alloc(	dev_A_temp_mat_of_num_K_roi_num_T,
				"dev_A_temp_mat_of_num_K_roi_num_T",
				num_K * roi_num_T);

	dev_alloc(	dev_B_temp_mat_of_num_K_roi_num_T,
				"dev_B_temp_mat_of_num_K_roi_num_T",
				num_K * roi_num_T);

	dev_alloc(	dev_C_temp_mat_of_num_K_roi_num_T,
				"dev_C_temp_mat_of_num_K_roi_num_T",
				num_K * roi_num_T);

	dev_alloc(	dev_A_temp_vec_of_num_K,
				"dev_A_temp_vec_of_num_K",
				num_K);

	dev_alloc(	dev_B_temp_vec_of_num_K,
				"dev_B_temp_vec_of_num_K",
				num_K);

}

void Vsr::sim_model(int random_seed, bool save_to_file) {

	// start of declarations

	int i, j;

	// end of declarations

	for (i = 0; i < num_T; i++) {

		models[i]->simulate(models, i, random_seed);

	}

	if (save_to_file) {
		for (i = 0; i < num_T; i++) {

			printf("[%d] (%d) printing to file...\n", random_seed, i);

			double *for_printing = new double[num_K * (i + 1)];

			dev_download(	for_printing,
							const_cast<double *>(models[i]->get_dev_ptr_to_col_of_p(0)),
							"dev_p",
							num_K * (i + 1));

			std::string file_name = std::to_string(random_seed) + "_" + std::to_string(i + 1) + ".csv";
			print_to_file(	"./output/" + file_name,
							for_printing,
							num_K,
							i + 1);

			delete[] for_printing;
		}
	}

	for (j = 0; j < num_pages; j++) {
		for (i = 0; i < roi_num_T; i++) {
			copy(	models[roi_left_T_index + i]->get_dev_ptr_to_col_of_p(j),
					dev_vec_p + IDXX(0, i, j, num_K, roi_num_T),
					num_K);
			copy(	models[roi_left_T_index + i]->get_dev_ptr_to_col_of_delta_p(j),
					dev_vec_delta_p + IDXX(0, i, j, num_K, roi_num_T),
					num_K);
		}
	}

	for (j = 0; j < num_pages; j++) {

		// dev_vec_1st_pd_of_p_wrt_T
		row_1st_pd(	dev_vec_p + IDXX(0, 0, j, num_K, roi_num_T),
					dev_vec_1st_pd_of_p_wrt_T + IDXX(0, 0, j, num_K, roi_num_T),
					num_K,
					roi_num_T,
					delta_T);

		// dev_vec_2nd_pd_of_p_wrt_T
		row_1st_pd(	dev_vec_1st_pd_of_p_wrt_T + IDXX(0, 0, j, num_K, roi_num_T),
					dev_vec_2nd_pd_of_p_wrt_T + IDXX(0, 0, j, num_K, roi_num_T),
					num_K,
					roi_num_T,
					delta_T);

		// dev_vec_3rd_pd_of_p_wrt_T
		row_1st_pd(	dev_vec_2nd_pd_of_p_wrt_T + IDXX(0, 0, j, num_K, roi_num_T),
					dev_vec_3rd_pd_of_p_wrt_T + IDXX(0, 0, j, num_K, roi_num_T),
					num_K,
					roi_num_T,
					delta_T);
	}

}

// public methods

Vsr::Vsr(	int num_K_val,
			double min_K_val,
			double max_K_val,
			int num_T_val,
			double min_T_val,
			double max_T_val,
			int roi_left_T_index_val,
			int roi_right_T_index_val,
			int device_index) {

	#if _DEBUG_VSR

		int device;
		cudaError = cudaGetDevice(&device);
		if (cudaError != cudaSuccess) {
			std::cout << "cudaGetDevice in Vsr constructor failed." << std::endl;
		}

		std::cout 	<< "VSR simulation at " << this
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
					<< " ROI range: [" << roi_left_T_index_val
					<< ", " << roi_right_T_index_val
					<< "] GPU: " << device
					<< " cuBlas: " << cublasHandle
					<< std::endl;

	#endif

	cublasStat = cublasCreate(&cublasHandle);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization failed.");
	}

	chosen_device_index = device_index;

	num_K = num_K_val;
	min_K = min_K_val;
	max_K = max_K_val;
	num_T = num_T_val;
	min_T = min_T_val;
	max_T = max_T_val;

	roi_left_T_index = roi_left_T_index_val;
	roi_right_T_index = roi_right_T_index_val;
	roi_num_T = roi_right_T_index - roi_left_T_index + 1;

	num_pages = roi_left_T_index + 1;

	delta_K	= ( (double) (max_K - min_K) ) / ( (double) (num_K - 1) );
	delta_T = ( (double) (max_T - min_T) ) / ( (double) (num_T - 1) );

	dev_Ks = NULL;
	host_Ks = new double[num_K];
	host_Ts = new double[num_T];

	// until here things will be initialized in Vsr()

	dev_forward_rate = NULL;

	// until here things will be initialized in init()

	// for simulation
	models = NULL;

	dev_vec_delta_p = NULL;
	dev_vec_p = NULL;

	dev_vec_1st_pd_of_p_wrt_T = NULL;
	dev_vec_2nd_pd_of_p_wrt_T = NULL;
	dev_vec_3rd_pd_of_p_wrt_T = NULL;

	host_delta_vsr = NULL;
	host_vsr = 0.0;

	// until here things will be initialized in alloc_sim_vars()

	// useful matrices

	dev_one_mat_num_T_num_T = NULL;
	dev_one_mat_num_K_num_T = NULL;

	dev_theta_1 = NULL;
	dev_theta_1_squared = NULL;
	dev_mu_delta_t = NULL;
	dev_lambda = NULL;

	// until here things will be initialized in init_useful_matrices()

	// temporary storage

	dev_A_temp_mat_of_num_K_roi_num_T = NULL;
	dev_B_temp_mat_of_num_K_roi_num_T = NULL;
	dev_C_temp_mat_of_num_K_roi_num_T = NULL;

	dev_A_temp_vec_of_num_K = NULL;
	dev_B_temp_vec_of_num_K = NULL;

	// until here things will be initialized in alloc_temp_vars()

	//================================================================

	// start of declarations

	int i;

	// end of declarations

	// initialize host_Ks and dev_Ks

	for (i = 0; i < num_K; i++) {
		host_Ks[i] = min_K + i * delta_K;
	}

	dev_alloc_and_init(	host_Ks,
						dev_Ks,
						"dev_Ks",
						num_K);

	// end of host_Ks and dev_Ks

	// initialize host_Ts

	for (i = 0; i < num_T; i++) {
		host_Ts[i] = min_T + i * delta_T;
	}

	// end of host_Ts

	// init matrices used for simulation

	this->alloc_temp_vars();
	this->alloc_sim_vars();

	// init useful matrices

	this->init_useful_matrices();

}

Vsr::~Vsr() {

	// destroy cublasHandle
	cublasStat = cublasDestroy(cublasHandle);
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS destroy failed.");
	}

	// strikes and maturities

	dev_release(	dev_Ks,
					"dev_Ks");
	delete[] host_Ks;
	delete[] host_Ts;

	// parameters

	dev_release(	dev_forward_rate,
					"dev_forward_rate");

	// for simulation
	for (int i = 0; i < num_T; i++)
		delete models[i];
	delete[] models;

	dev_release(	dev_vec_delta_p,
					"dev_vec_delta_p");
	dev_release(	dev_vec_p,
					"dev_vec_p");

	dev_release(	dev_vec_1st_pd_of_p_wrt_T,
					"dev_vec_1st_pd_of_p_wrt_T");
	dev_release(	dev_vec_2nd_pd_of_p_wrt_T,
					"dev_vec_2nd_pd_of_p_wrt_T");
	dev_release(	dev_vec_3rd_pd_of_p_wrt_T,
					"dev_vec_3rd_pd_of_p_wrt_T");

	delete[] host_delta_vsr;

	// useful matrices

	dev_release(	dev_one_mat_num_T_num_T,
					"dev_one_mat_num_T_num_T");
	dev_release(	dev_one_mat_num_K_num_T,
					"dev_one_mat_num_K_num_T");

	dev_release(	dev_theta_1,
					"dev_theta_1");
	dev_release(	dev_theta_1_squared,
					"dev_theta_1_squared");
	dev_release(	dev_mu_delta_t,
					"dev_mu_delta_t");
	dev_release(	dev_lambda,
					"dev_lambda");

	// temporary storage

	dev_release(	dev_A_temp_mat_of_num_K_roi_num_T,
					"dev_A_temp_mat_of_num_K_roi_num_T");
	dev_release(	dev_B_temp_mat_of_num_K_roi_num_T,
					"dev_B_temp_mat_of_num_K_roi_num_T");
	dev_release(	dev_C_temp_mat_of_num_K_roi_num_T,
					"dev_C_temp_mat_of_num_K_roi_num_T");

	dev_release(	dev_A_temp_vec_of_num_K,
					"dev_A_temp_vec_of_num_K");
	dev_release(	dev_B_temp_vec_of_num_K,
					"dev_B_temp_vec_of_num_K");

}

void Vsr::init(	const double *host_input_mu_k,
				const double *host_input_sigma_k,
				const double *host_input_sigma_f,
				const double *host_input_q_0,
				const double *host_input_cor_w,
				const double host_input_c) {
	/*
	 * Assumptions:
	 * 1.	mu_k is both t-varying and tau-varying
	 * 2.	sigma_k and sigma_f are tau-varying but not t-varying
	 * 3.	mu_f is both t-varying and tau-varying decided when running
	 * 4.	host_input_cor_w is not t-varying
	 *
	 * So:
	 * 1.	mu_k should be a double array of the length num_T * num_T
	 * 		in the order of
	 * 		{{mu_k(t, tau_0)}, {mu_k(t, tau_1)}, ..., {mu_k(t, tau_num_T)}},
	 * 		where {mu_k(t, tau_i)} =
	 * 		{mu_k(t_0, tau_i), mu_k(t_1, tau_i), ..., mu_k(t_num_T, tau_i)}.
	 * 2.	sigma_k simply is a double arrays of the length
	 * 		num_T like {sigma_f(tau_0), ..., sigma_f(tau_num_T)}.
	 * 3.	sigma_f is an array of the length num_K * num_T like
	 * 		{{sigma_f(k, tau_0)}, {sigma_f(k, tau_1)}, ...,
	 * 		{sigma_f(k, tau_num_T)}}, where {sigma_f(k, tau_i)} =
	 * 		{sigma_f(k_0, tau_i), sigma_f(k_1, tau_i), ...,
	 * 		sigma_f(k_num_K, tau_i)}.
	 * 4.	In the above setting, tau_0 < tau_1 < ... < tau_num_T.
	 *
	 */

	// start of declarations

	int i, j, k, new_num_T;
	double *host_mu_k, *host_sigma_k, *host_sigma_f, *host_q_0;

	// end of declarations

	dev_alloc_and_init(	host_input_mu_k + roi_left_T_index,
						dev_forward_rate,
						"dev_forward_rate",
						roi_num_T);

	// initialize models using the above parameters

	for (k = 0; k < num_T; k++) {

		new_num_T = 1 + k;

		host_mu_k = new double[new_num_T * new_num_T];
		for (j = 0; j < new_num_T; j++)
			for (i = 0; i < new_num_T; i++)
				// reverse to tau_0 > tau_1 > ... > tau_num_T
				host_mu_k[IDX(i, j, new_num_T)] =
						host_input_mu_k[IDX(i, new_num_T - 1 - j, num_T)];

		host_sigma_k = new double[new_num_T];
		for (i = 0; i < new_num_T; i++)
			// reverse to tau_0 > tau_1 > ... > tau_num_T
			host_sigma_k[i] =
					host_input_sigma_k[new_num_T - 1 - i];

		host_sigma_f = new double[num_K * new_num_T];
		for (j = 0; j < new_num_T; j++)
			for (i = 0; i < num_K; i++)
				// reverse to tau_0 > tau_1 > ... > tau_num_T
				host_sigma_f[IDX(i, j, num_K)] =
						host_input_sigma_f[IDX(i, new_num_T - 1 - j, num_K)];

		host_q_0 = new double[num_K * new_num_T];
		for (j = 0; j < new_num_T; j++)
			for (i = 0; i < num_K; i++)
				// reverse to tau_0 > tau_1 > ... > tau_num_T
				host_q_0[IDX(i, j, num_K)] =
						host_input_q_0[IDX(i, new_num_T - 1 - j, num_K)];

		models[k]->init(	host_mu_k,
							host_sigma_k,
							host_sigma_f,
							host_q_0,
							host_input_cor_w,
							host_input_c);

		// delete temporary
		delete[] host_mu_k;
		delete[] host_sigma_k;
		delete[] host_sigma_f;
		delete[] host_q_0;

	}

}

double Vsr::sim_vsr(double init_vsr_val, int random_seed, bool save_to_file) {

	// start of declarations

	double alpha, beta, adj = 6.5;
	double 	*dev_ptr_to_1st_pd,
			*dev_ptr_to_2nd_pd,
			*dev_ptr_to_3rd_pd,
			*dev_ptr_to_delta_p;

	// end of declarations

	host_vsr = init_vsr_val / adj;

	this->sim_model(random_seed, save_to_file);

	for (int j = 0; j < (num_pages - 1); j++) {

		dev_ptr_to_1st_pd = dev_vec_1st_pd_of_p_wrt_T + IDXX(0, 0, j, num_K, roi_num_T);
		dev_ptr_to_2nd_pd = dev_vec_2nd_pd_of_p_wrt_T + IDXX(0, 0, j, num_K, roi_num_T);
		dev_ptr_to_3rd_pd = dev_vec_3rd_pd_of_p_wrt_T + IDXX(0, 0, j, num_K, roi_num_T);
		dev_ptr_to_delta_p = dev_vec_delta_p + IDXX(0, 0, j, num_K, roi_num_T);

		// beta for theta
		elementwise_vec_prod(	dev_ptr_to_3rd_pd,
								dev_ptr_to_1st_pd,
								dev_A_temp_mat_of_num_K_roi_num_T,
								num_K * roi_num_T);
		vec_squaring(	dev_ptr_to_2nd_pd,
						dev_B_temp_mat_of_num_K_roi_num_T,
						num_K * roi_num_T);
		elementwise_vec_sub(	dev_A_temp_mat_of_num_K_roi_num_T,
								dev_B_temp_mat_of_num_K_roi_num_T,
								dev_C_temp_mat_of_num_K_roi_num_T,
								num_K * roi_num_T);

		vec_cubing(	dev_ptr_to_1st_pd,
					dev_A_temp_mat_of_num_K_roi_num_T,
					num_K * roi_num_T);
		elementwise_vec_div(	dev_C_temp_mat_of_num_K_roi_num_T,
								dev_A_temp_mat_of_num_K_roi_num_T,
								dev_B_temp_mat_of_num_K_roi_num_T,
								num_K * roi_num_T);

		vec_squaring(	dev_ptr_to_delta_p,
						dev_A_temp_mat_of_num_K_roi_num_T,
						num_K * roi_num_T);
		elementwise_vec_prod(	dev_A_temp_mat_of_num_K_roi_num_T,
								dev_B_temp_mat_of_num_K_roi_num_T,
								dev_C_temp_mat_of_num_K_roi_num_T,					// beta
								num_K * roi_num_T);

		// alpha for theta
		elementwise_vec_div(	dev_ptr_to_2nd_pd,
								dev_ptr_to_1st_pd,
								dev_A_temp_mat_of_num_K_roi_num_T,
								num_K * roi_num_T);
		elementwise_vec_prod(	dev_A_temp_mat_of_num_K_roi_num_T,
								dev_ptr_to_delta_p,
								dev_B_temp_mat_of_num_K_roi_num_T,
								num_K * roi_num_T);
		vec_scaling(	dev_B_temp_mat_of_num_K_roi_num_T,
						dev_A_temp_mat_of_num_K_roi_num_T,							// alpha
						num_K * roi_num_T,
						2.0);

		// alpha + beta
		elementwise_vec_sum(	dev_C_temp_mat_of_num_K_roi_num_T,
								dev_A_temp_mat_of_num_K_roi_num_T,
								dev_B_temp_mat_of_num_K_roi_num_T,					// alpha + beta
								num_K * roi_num_T);

		// gamma
		// 2 will be multiplied in the following
		rowwise_mat_prod_with_col_vec(	dev_ptr_to_delta_p,
										dev_Ks,
										dev_A_temp_mat_of_num_K_roi_num_T,
										num_K,
										roi_num_T);
		colwise_mat_prod_with_row_vec(	dev_A_temp_mat_of_num_K_roi_num_T,
										dev_forward_rate,
										dev_C_temp_mat_of_num_K_roi_num_T,			// gamma
										num_K,
										roi_num_T);

		// alpha * lambda + beta * lambda

		alpha = 1.0;
		beta = 0.0;
		cublasStat = cublasDgemm(	cublasHandle,
									CUBLAS_OP_N, CUBLAS_OP_N,
									num_K, roi_num_T, num_K,
									&alpha,
									dev_theta_1_squared, num_K,
									dev_B_temp_mat_of_num_K_roi_num_T, num_K,
									&beta,
									dev_A_temp_mat_of_num_K_roi_num_T, num_K);
		if (cublasStat != CUBLAS_STATUS_SUCCESS) {
			printf("cublasDgemm in sim_vsr failed.");
		}

		alpha = 2.0;
		beta = 1.0;
		cublasStat = cublasDgemm(	cublasHandle,
									CUBLAS_OP_N, CUBLAS_OP_N,
									num_K, roi_num_T, num_K,
									&alpha,
									dev_theta_1, num_K,
									dev_C_temp_mat_of_num_K_roi_num_T, num_K,
									&beta,
									dev_A_temp_mat_of_num_K_roi_num_T, num_K);
		if (cublasStat != CUBLAS_STATUS_SUCCESS) {
			printf("cublasDgemm in sim_vsr failed.");
		}

		alpha = 1.0;
		beta = 0.0;
		cublasStat = cublasDgemv(	cublasHandle,
									CUBLAS_OP_N,
									num_K, roi_num_T,
									&alpha,
									dev_A_temp_mat_of_num_K_roi_num_T, num_K,
									dev_mu_delta_t, 1,
									&beta,
									dev_A_temp_vec_of_num_K, 1);
		if (cublasStat != CUBLAS_STATUS_SUCCESS) {
			printf("cublasDgemv in sim_vsr failed.");
		}

		elementwise_vec_prod(	dev_A_temp_vec_of_num_K,
								dev_lambda,
								dev_B_temp_vec_of_num_K,
								num_K);

		vec_sum(	cublasHandle,
					dev_B_temp_vec_of_num_K,
					host_delta_vsr + j,
					num_K);

		host_vsr += host_delta_vsr[j] / (roi_num_T * delta_T);

	}

//	printf("%p : %f\n", this, host_vsr * 100.0);

	return host_vsr * adj;

}

