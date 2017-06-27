//
// Created by xixuan on 4/3/17.
//

#ifndef EVAL_H
#define EVAL_H

#include "model.h"
#include "vsr.h"

#ifndef _OMP
	#define _OMP
#endif

#define _MAX_RAND_SEED 1000000

#define IDX(i, j, ld_i) (((j)*(ld_i))+(i))
#define IDXX(i, j, k, ld_i, ld_j) (((k)*((ld_i)*(ld_j)))+((j)*(ld_i))+(i))

class Derivative {

private:

	double const_mu_k;
	double const_sigma_k;

protected:

	double slop_sigma_f;
	double corr_coef;

	// for evaluation
	int num_threads;
	int *gpu_indices;

	int num_evals_in_each_thread;
	int *random_seeds;
	double *vsr_array;
	double *discounted_payoff_array;

	// strikes
	int num_K;
	double min_K;
	double max_K;
	double delta_K;

	// maturities
	int num_T;
	double min_T;
	double max_T;
	double delta_T;

	int roi_left_T_index;								// start from 0
	int roi_right_T_index;								// start from 0

	Vsr **vsrs;

	// initial surface
	int num_q_0_taus;
	double *q_0_taus;
	double *q_0;

	// parameter c
	double c;

	// results
	double average;
	double std_dev;

	// for internal uses
	virtual double fun_mu_k(double t, double tau);
	virtual double fun_sigma_k(double tau);
	virtual double fun_sigma_f(double k, double tau);
	virtual double fun_corr(double k_0, double k_1);
	virtual double fun_q_0(double k, double tau);

	virtual double payoff(double vsr);
	virtual double discounting();

public:

	Derivative(	int num_K_val,
				double min_K_val,
				double max_K_val,

				int num_T_val,
				double min_T_val,
				double max_T_val,

				int roi_left_T_index_val,
				int roi_right_T_index_val,

				int num_threads_val,
				int const * gpu_indices_val,

				int num_evals_in_each_thread_val,

				int input_num_q_0_taus,
				double const *input_q_0_taus,
				double const *input_q_0);

	virtual ~Derivative();

	void set_params(double const_mu_k_val,
					double const_sigma_k_val,
					double slop_sigma_f_val,
					double corr_coef_val,
					double c_val);

	double evaluate(double init_vsr);

	double get_sd();

};

class VixOpt : public Derivative {

private:

	double slop_mu_k;

	int num_rf_rates;
	double *rf_rates_taus;
	double *rf_rates;

	double slop_sigma_k;
	double intercept_sigma_k;

	char opt_type;
	double strike;

	int num_div_rates;
	double *div_rates_taus;
	double *div_rates;

protected:

	double fun_rf_rate(double tau);

	double fun_mu_k(double t, double tau);
	double fun_sigma_k(double tau);

	double payoff(double vsr);
	double discounting();

public:

	VixOpt(	int num_K_val,
			double min_K_val,
			double max_K_val,

			int num_T_val,
			double min_T_val,
			double max_T_val,

			int roi_left_T_index_val,
			int roi_right_T_index_val,

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
			double const *div_rates_val);

	virtual ~VixOpt();

	void simulate(	double slop_mu_k_val,
					double slop_sigma_k_val,
					double intercept_sigma_k_val,
					double slop_sigma_f_val,
					double corr_coef_val,

					double c_val,
					double init_vsr,

					bool save_to_file = false);

	double evaluate(char opt_type_val, double strike_val);

};


#endif //EVAL_H
