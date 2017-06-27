//
// Created by xixuan on 4/3/17.
//
// Note: everything is column majored.
//

#include "optimization.h"

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <algorithm>

bool compare(params_mspe const &left, params_mspe const &right) {
	return left.mspe < right.mspe;
}

double Optim::get_init_vsr(double tau) {

	double init_vsr;

	for (int i = 0; i < num_init_vsrs_taus + 1; i++) {
		if (i != num_init_vsrs_taus && tau < init_vsrs_taus[i]) {
			if (i > 0)
				init_vsr =	( (init_vsrs_taus[i] - tau) * init_vsrs[i - 1] +
							(tau - init_vsrs_taus[i - 1]) * init_vsrs[i] ) /
							(init_vsrs_taus[i] - init_vsrs_taus[i - 1]);
			else
				init_vsr = init_vsrs[0];
			break;
		}
		if (i == num_init_vsrs_taus) {
			init_vsr = init_vsrs[num_init_vsrs_taus - 1];
		}
	}

	return init_vsr;

}

Optim::Optim(	int num_K_val,
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
				double *init_vsrs_val) {

	num_slop_mu_k_candidates = 2 * num_smk_candidates_on_a_side + 1;
	slop_mu_k_candidates = new double[num_slop_mu_k_candidates];

	num_slop_sigma_k_candidates = 2 * num_ssk_candidates_on_a_side + 1;
	slop_sigma_k_candidates = new double[num_slop_sigma_k_candidates];

	num_intercept_sigma_k_candidates = 2 * num_isk_candidates_on_a_side + 1;
	intercept_sigma_k_candidates = new double[num_intercept_sigma_k_candidates];

	num_slop_sigma_f_candidates = 2 * num_ssf_candidates_on_a_side + 1;
	slop_sigma_f_candidates = new double[num_slop_sigma_f_candidates];

	num_corr_coef_candidates = 2 * num_cc_candidates_on_a_side + 1;
	corr_coef_candidates = new double[num_corr_coef_candidates];

	num_strikes = num_strikes_val;
	num_maturities = num_maturities_val;
	strikes = new double[num_strikes * num_maturities];
	maturities = new double[num_maturities];
	prices = new double[num_strikes * num_maturities];

	for (int i = 0; i < num_strikes * num_maturities; i++) {
		strikes[i] = strikes_val[i];
		prices[i] = prices_val[i];
	}
	for (int i = 0; i < num_maturities; i++)
		maturities[i] =
				( ( (double) (max_T_val[i] - min_T_val[i]) ) / ( (double) (num_T_val[i] - 1) ) *
						roi_left_T_index_val[i] + min_T_val[i]);

	num_init_vsrs_taus = num_init_vsrs_taus_val;
	init_vsrs_taus = new double[num_init_vsrs_taus];
	init_vsrs = new double[num_init_vsrs_taus];
	for (int i = 0; i < num_init_vsrs_taus; i++) {
		init_vsrs_taus[i] = init_vsrs_taus_val[i];
		init_vsrs[i] = init_vsrs_val[i];
	}

	num_mspes = 	num_slop_mu_k_candidates *
					num_slop_sigma_k_candidates *
					num_intercept_sigma_k_candidates *
					num_slop_sigma_f_candidates *
					num_corr_coef_candidates;
	results = new params_mspe[num_mspes];

	options = new VixOpt*[num_maturities];
	for (int i = 0; i < num_maturities; i++) {
		options[i] = new VixOpt(num_K_val,
								min_K_val,
								max_K_val,

								num_T_val[i],
								min_T_val[i],
								max_T_val[i],

								roi_left_T_index_val[i],
								roi_right_T_index_val[i],

								num_threads_val,
								gpu_indices_val,

								num_evals_in_each_thread_val,

								input_num_q_0_taus,
								input_q_0_taus,
								input_q_0,

								num_rf_rates_val,
								rf_rates_taus_val,
								rf_rates_val,

								num_div_rates_val,
								div_rates_taus_val,
								div_rates_val);
	}

}

Optim::~Optim() {

	delete[] slop_mu_k_candidates;
	delete[] slop_sigma_k_candidates;
	delete[] intercept_sigma_k_candidates;
	delete[] slop_sigma_f_candidates;
	delete[] corr_coef_candidates;

	delete[] strikes;
	delete[] maturities;
	delete[] prices;

	delete[] init_vsrs_taus;
	delete[] init_vsrs;

	delete[] results;

	for (int i = 0; i < num_maturities; i++)
		delete options[i];
	delete[] options;

}

void Optim::set_param_candidates(	double center_slop_mu_k,
									double radius_slop_mu_k,

									double center_slop_sigma_k,
									double radius_slop_sigma_k,

									double center_intercept_sigma_k,
									double radius_intercept_sigma_k,

									double center_slop_sigma_f,
									double radius_slop_sigma_f,

									double center_corr_coef,
									double radius_corr_coef) {

	double delta;
	int i, num_candidates_on_a_side;

	num_candidates_on_a_side = ((num_slop_mu_k_candidates - 1) / 2);
	delta = radius_slop_mu_k / (num_candidates_on_a_side == 0 ? 1.0 : num_candidates_on_a_side);
	for (i = 0; i < num_slop_mu_k_candidates; i++)
		slop_mu_k_candidates[i] = center_slop_mu_k - delta * (num_candidates_on_a_side - i);

	num_candidates_on_a_side = ((num_slop_sigma_k_candidates - 1) / 2);
	delta = radius_slop_sigma_k / (num_candidates_on_a_side == 0 ? 1.0 : num_candidates_on_a_side);
	for (i = 0; i < num_slop_sigma_k_candidates; i++)
		slop_sigma_k_candidates[i] = center_slop_sigma_k - delta * (num_candidates_on_a_side - i);

	num_candidates_on_a_side = ((num_intercept_sigma_k_candidates - 1) / 2);
	delta = radius_intercept_sigma_k / (num_candidates_on_a_side == 0 ? 1.0 : num_candidates_on_a_side);
	for (i = 0; i < num_intercept_sigma_k_candidates; i++)
		intercept_sigma_k_candidates[i] = center_intercept_sigma_k - delta * (num_candidates_on_a_side - i);

	num_candidates_on_a_side = ((num_slop_sigma_f_candidates - 1) / 2);
	delta = radius_slop_sigma_f / (num_candidates_on_a_side == 0 ? 1.0 : num_candidates_on_a_side);
	for (i = 0; i < num_slop_sigma_f_candidates; i++)
		slop_sigma_f_candidates[i] = center_slop_sigma_f - delta * (num_candidates_on_a_side - i);

	num_candidates_on_a_side = ((num_corr_coef_candidates - 1) / 2);
	delta = radius_corr_coef / (num_candidates_on_a_side == 0 ? 1.0 : num_candidates_on_a_side);
	for (i = 0; i < num_corr_coef_candidates; i++)
		corr_coef_candidates[i] = center_corr_coef - delta * (num_candidates_on_a_side - i);

}

void Optim::optimize(	char opt_type_val,
						double c_val,
						int random_seed) {

	int index = 0;
	double price, spe;

	for (int i = 0; i < num_slop_mu_k_candidates; i++) {
		for (int j = 0; j < num_slop_sigma_k_candidates; j++) {
			for (int k = 0; k < num_intercept_sigma_k_candidates; k++) {
				for (int l = 0; l < num_slop_sigma_f_candidates; l++) {
					for (int m = 0; m < num_corr_coef_candidates; m++) {

						printf("\n<<%d>> Optimizing for "
								"slop_mu_k %f "
								"slop_sigma_k %f "
								"intercept_sigma_k %f "
								"slop_sigma_f %f "
								"corr_coef %f \n",
								index,
								slop_mu_k_candidates[i],
								slop_sigma_k_candidates[j],
								intercept_sigma_k_candidates[k],
								slop_sigma_f_candidates[l],
								corr_coef_candidates[m]);

						printf("Initial VSR: ");
						for (int n = 0; n < num_maturities; n++) {

							printf("(%d) %f ", n, this->get_init_vsr(maturities[n]));

							srand(random_seed);

							options[n]->simulate(	slop_mu_k_candidates[i],
													slop_sigma_k_candidates[j],
													intercept_sigma_k_candidates[k],
													slop_sigma_f_candidates[l],
													corr_coef_candidates[m],
													c_val,
													this->get_init_vsr(maturities[n]));

						}
						printf("\n");

						results[index].slop_mu_k = slop_mu_k_candidates[i];
						results[index].slop_sigma_k = slop_sigma_k_candidates[j];
						results[index].intercept_sigma_k = intercept_sigma_k_candidates[k];
						results[index].slop_sigma_f = slop_sigma_f_candidates[l];
						results[index].corr_coef = corr_coef_candidates[m];

						results[index].mspe = 0.0;
						for (int n = 0; n < num_maturities; n++) {
							for (int p = 0; p < num_strikes; p++) {

								price = options[n]->evaluate(	opt_type_val,
																strikes[IDX(p, n, num_strikes)]);

								spe = 	(prices[IDX(p, n, num_strikes)] - price) *
										(prices[IDX(p, n, num_strikes)] - price) /
										prices[IDX(p, n, num_strikes)] /
										prices[IDX(p, n, num_strikes)];
								results[index].mspe += spe;

								printf(	"Maturity: %.4f "
										"Strike: %.4f "
										"Price: %.4f "
										"Model: %.4f "
										"SPE: %.4f \n",
										maturities[n],
										strikes[p],
										prices[IDX(p, n, num_strikes)],
										price,
										spe);

							}
						}

						results[index].mspe /= (double) (num_strikes * num_maturities);

						printf("<< summary >> MSPE: %f\n", results[index].mspe);

						index += 1;

					}
				}
			}
		}
	}

	std::sort(results, results + num_mspes, compare);

}

params_mspe const *Optim::get_sorted_mspe(int index) const {
	/*
	 * index = 0 gives the smallest mspe.
	 */

	return results + index;

}
