//
// Created by xixuan on 4/3/17.
//

#ifndef OPTIM_H
#define OPTIM_H

#include "evaluation.h"
#include "tools.h"

#define IDX(i, j, ld_i) (((j)*(ld_i))+(i))
#define IDXX(i, j, k, ld_i, ld_j) (((k)*((ld_i)*(ld_j)))+((j)*(ld_i))+(i))

struct params_mspe {

	double slop_mu_k;
	double slop_sigma_k;
	double intercept_sigma_k;
	double slop_sigma_f;
	double corr_coef;
	double mspe;

};

class Optim {

private:

	int num_maturities;

	int num_slop_mu_k_candidates;
	double *slop_mu_k_candidates;

	int num_slop_sigma_k_candidates;
	double *slop_sigma_k_candidates;

	int num_intercept_sigma_k_candidates;
	double *intercept_sigma_k_candidates;

	int num_slop_sigma_f_candidates;
	double *slop_sigma_f_candidates;

	int num_corr_coef_candidates;
	double *corr_coef_candidates;

	int num_strikes;
	double *strikes;
	double *maturities;
	double *prices;

	int num_init_vsrs_taus;
	double *init_vsrs_taus;
	double *init_vsrs;

	int num_mspes;
	params_mspe *results;

	VixOpt **options;

	double get_init_vsr(double tau);

public:

	Optim(	int num_K_val,
			double min_K_val,
			double max_K_val,

			int const *num_T_val,
			double const *min_T_val,
			double const *max_T_val,

			int const *roi_left_T_index_val,
			int const *roi_right_T_index_val,

			int num_threads_val,
			int const * gpu_indices_val,
			int num_evals_in_each_thread_val,

			int input_num_q_0_taus,
			double const *input_q_0_taus,
			double const *input_q_0,

			int num_rf_rates_val,
			double const *rf_rates_taus_val,
			double const *rf_rates_val,

			int num_div_rates_val,
			double const *div_rates_taus_val,
			double const *div_rates_val,

			int num_smk_candidates_on_a_side,
			int num_ssk_candidates_on_a_side,
			int num_isk_candidates_on_a_side,
			int num_ssf_candidates_on_a_side,
			int num_cc_candidates_on_a_side,

			int num_strikes_val,
			int num_maturities_val,
			double const *strikes_val,
			double const *prices_val,

			int num_init_vsrs_taus_val,
			double *init_vsrs_taus_val,
			double *init_vsrs_val);
	~Optim();

	void set_param_candidates(	double center_slop_mu_k,
								double radius_slop_mu_k,

								double center_slop_sigma_k,
								double radius_slop_sigma_k,

								double center_intercept_sigma_k,
								double radius_intercept_sigma_k,

								double center_slop_sigma_f,
								double radius_slop_sigma_f,

								double center_corr_coef,
								double radius_corr_coef);

	void optimize(	char opt_type_val,
					double c_val,
					int random_seed);

	params_mspe const *get_sorted_mspe(int index) const;

};

#endif //OPTIM_H
