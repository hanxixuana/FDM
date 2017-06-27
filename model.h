//
// Created by xixuan on 4/3/17.
//

#ifndef MODEL_H
#define MODEL_H

#include <cublas_v2.h>

#define _DEBUG_MODEL 0
#define IDX(i, j, ld_i) (((j)*(ld_i))+(i))
#define IDXX(i, j, k, ld_i, ld_j) (((k)*((ld_i)*(ld_j)))+((j)*(ld_i))+(i))

class Sim {

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

	// strikes
	int num_K;
	double min_K;
	double max_K;

	// maturities
	int num_T;
	double min_T;
	double max_T;

	// delta t
	double delta_K;
	double delta_T;

	// strikes and maturities (stored on device)
	double *dev_Ks;										// by Sim()
	double *dev_Ts;										// by Sim()

	// random factors (stored on device)
    double *dev_delta_b;									// alloc by alloc_sim_vars()
    													// sim by deltas()
    double *dev_delta_w;									// alloc by alloc_sim_vars()
														// sim by deltas()

	// parameters (stored on device)
	double *dev_mu_k;									// by init()
	double *dev_sigma_k;									// by init()
	double *dev_sigma_f;									// by init()
	double *dev_q_0;										// by init()
	double *dev_cor_w;									// by init()
														// and does not support t-varying correlation

	// parameters (stored on host)
	double *host_cor_w;									// by init()
	double host_c;										// by init()

	// for simulation
	double *dev_mu_f;									// alloc by alloc_sim_vars()
														// init by init_mu_f()
														// sim by sim_mu_f()
	double *dev_all_mu_f;

	double *dev_shifted_q_0;								// by generate_k_proc()
	double *dev_1st_pd_of_shifted_q_0_wrt_K_over_q_0;	// by generate_k_proc()
	double *dev_1st_pd_of_shifted_q_0_wrt_T_over_q_0;	// by generate_k_proc()
	double *dev_2nd_pd_of_shifted_q_0_wrt_K_over_q_0;	// by generate_k_proc()

	double *dev_k;										// by generate_k_proc()
	double *dev_1st_pd_of_k_wrt_T;						// by generate_k_proc()

	double *dev_f;										// by sim_f()
	double *dev_1st_pd_of_f_wrt_T_over_f;				// by sim_f()

	double *dev_delta_p;									// by simulate()
	double *dev_p;										// by sim_p()

	double *nt_mat_w;
	double *nt_mat_b;

	// useful matrices

	double *dev_one_mat_num_T_num_T;
	double *dev_one_mat_num_K_num_T;

	double *dev_zero_vec_num_K;

	// temporary storage

	double *dev_A_temp_mat_of_num_K_num_K;
	double *dev_A_temp_mat_of_num_K_num_T;
	double *dev_A_temp_mat_of_num_T_num_T;
	double *dev_A_temp_vec_of_num_K;
	double *dev_A_temp_vec_of_num_T;

	double *dev_B_temp_mat_of_num_K_num_T;
	double *dev_B_temp_mat_of_num_T_num_T;
	double *dev_B_temp_vec_of_num_K;
	double *dev_B_temp_vec_of_num_T;

	double *dev_C_temp_mat_of_num_K_num_T;
	double *dev_C_temp_mat_of_num_T_num_T;
	double *dev_C_temp_vec_of_num_K;
	double *dev_C_temp_vec_of_num_T;

	/* ===========================
	 * 		private methods
	 * ===========================
	 */

	// initialization-related methods

	void init_useful_matrices();
	void alloc_sim_vars();
	void alloc_temp_vars();

	// simulation

	void deltas(int random_seed = 1,
				bool show_summary = false);
	void sim_k();
	void init_mu_f();

	void sim_mu_f(int col_index);
	void sim_f(	Sim ** const &models,
				int model_index,
				int col_index);
	void sim_p(int col_index);

public:

	Sim(	int num_K_val,
    		double min_K_val,
    		double max_K_val,
    		int num_T_val,
    		double min_T_val,
    		double max_T_val,
    		int device_index = 0);

    ~Sim();

    void init(	const double *host_mu_k,
				const double *host_sigma_k,
				const double *host_sigma_f,
				const double *host_q_0,
				const double *host_input_cor_w,
				const double c);

    void simulate(	Sim ** const &models,
    				int model_index,
    				int random_seed = 1);

	double const *get_dev_ptr_to_col_of_p(const int col_index) const;
	double const *get_dev_ptr_to_col_of_delta_p(const int col_index) const;

	double const *get_dev_ptr_to_col_of_mu_f(const int col_index) const;

};


#endif //MODEL_H
