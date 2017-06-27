//
// Created by xixuan on 4/3/17.
//

#ifndef VSR_H
#define VSR_H

#include "model.h"
#include <cublas_v2.h>

#define _PRINT_TO_FILE

#define _DEBUG_VSR 0
#define IDX(i, j, ld_i) (((j)*(ld_i))+(i))
#define IDXX(i, j, k, ld_i, ld_j) (((k)*((ld_i)*(ld_j)))+((j)*(ld_i))+(i))

class Vsr {

private:

	/* ===========================
	 * 		private variables
	 * ===========================
	 */

	// cublasHandle
	cublasHandle_t cublasHandle;

	// cuda and cublas return status
	cudaError_t cudaError;
	cublasStatus_t cublasStat;

	int chosen_device_index;

	// strikes
	int num_K;
	double min_K;
	double max_K;

	// maturities
	int num_T;
	double min_T;
	double max_T;

	int roi_left_T_index;								// start from 0
	int roi_right_T_index;								// start from 0
	int roi_num_T;

	int num_pages;

	// delta t
	double delta_K;
	double delta_T;

	// strikes and maturities (stored on device)
	double *dev_Ks;										// by Sim()
	double *host_Ks;									// by Sim()
	double *host_Ts;									// by Sim()

	// parameters (stored on device)
	double *dev_forward_rate;							// by init()

	// for simulation

	Sim **models;										// by alloc_sim_vars and sim_model()

	// num_K * roi_num_T * num_pages
	double *dev_vec_delta_p;							// by alloc_sim_vars and sim_model()
	double *dev_vec_p;									// by alloc_sim_vars and sim_model()

	double *dev_vec_1st_pd_of_p_wrt_T;					// by alloc_sim_vars and sim_model()
	double *dev_vec_2nd_pd_of_p_wrt_T;					// by alloc_sim_vars and sim_model()
	double *dev_vec_3rd_pd_of_p_wrt_T;					// by alloc_sim_vars and sim_model()

	// roi_left_T_index
	double *host_delta_vsr;								// by alloc_sim_vars and sim_vsr()
	double host_vsr;									// by alloc_sim_vars and sim_vsr()

	// useful matrices

	double *dev_one_mat_num_T_num_T;
	double *dev_one_mat_num_K_num_T;

	double *dev_theta_1;
	double *dev_theta_1_squared;
	double *dev_mu_delta_t;
	double *dev_lambda;

	// temporary storage

	double *dev_A_temp_mat_of_num_K_roi_num_T;
	double *dev_B_temp_mat_of_num_K_roi_num_T;
	double *dev_C_temp_mat_of_num_K_roi_num_T;

	double *dev_A_temp_vec_of_num_K;
	double *dev_B_temp_vec_of_num_K;

	/* ===========================
	 * 		private methods
	 * ===========================
	 */

	// initialization-related methods

	void init_useful_matrices();
	void alloc_sim_vars();
	void alloc_temp_vars();

	// simulation

	void sim_model(int random_sed, bool save_to_file = false);

public:

	Vsr(	int num_K_val,
    		double min_K_val,
    		double max_K_val,

    		int num_T_val,
    		double min_T_val,
    		double max_T_val,

    		int roi_left_T_index_val,
    		int roi_right_T_index_val,
    		int device_index = 0);

    ~Vsr();

    void init(	const double *host_input_mu_k,
				const double *host_input_sigma_k,
				const double *host_input_sigma_f,
				const double *host_input_q_0,
				const double *host_input_cor_w,
				const double host_input_c);

	double sim_vsr(	double init_vsr_val,
					int random_seed,
					bool save_to_file = false);

};


#endif //VSR_H
