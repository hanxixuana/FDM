//
// Created by xixuan on 4/3/17.
//
// Note: everything is column majored.
//

#include "evaluation.h"

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>

#ifdef _OMP
	#include <omp.h>
#endif

/*
 * 		Derivative
 */

double Derivative::fun_mu_k(double t, double tau) {

	return const_mu_k;
}
double Derivative::fun_sigma_k(double tau) {

	return const_sigma_k;
}
double Derivative::fun_sigma_f(double k, double tau) {

	return slop_sigma_f * tau;
}

double Derivative::fun_corr(double k_0, double k_1) {
	return std::exp( corr_coef * std::abs(k_0 - k_1) );
}

double Derivative::fun_q_0(double k, double tau) {

	int k_index = std::round((k - min_K) / delta_K);
	double result;

	for (int i = 0; i < num_q_0_taus + 1; i++) {
		if (i != num_q_0_taus && tau < q_0_taus[i]) {
			if (i > 0)
				result =	( (q_0_taus[i] - tau) * q_0[IDX(k_index, i - 1, num_K)] +
							(tau - q_0_taus[i - 1]) * q_0[IDX(k_index, i, num_K)] ) /
							(q_0_taus[i] - q_0_taus[i - 1]);
			else
				result = q_0[IDX(k_index, 0, num_K)];
			break;
		}
		if (i == num_q_0_taus)
			result = q_0[IDX(k_index, num_q_0_taus - 1, num_K)];
	}

	return result;

}

double Derivative::payoff(double vsr) {

	return vsr;
}
double Derivative::discounting() {

//	double discounting_factor = 1.0;
//	for (int i = 0; i < roi_left_T_index; i++)
//		discounting_factor *= std::exp( - this->fun_mu_k(0, i * delta_T) * delta_T );
//
//	return discounting_factor;

	return 1.0;
}

Derivative::Derivative(	int num_K_val,
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
						double const *input_q_0) {

	const_mu_k = 0.0;
	const_sigma_k = 0.0;
	slop_sigma_f = 0.0;
	corr_coef = -1.0;

	#ifdef _OMP
		num_threads = num_threads_val;
	#else
		num_threads = 1;
		printf("OMP has been disabled and num_threads is set to 1 instead.");
	#endif
	gpu_indices = new int[num_threads];
	for (int i = 0; i < num_threads; i++)
		gpu_indices[i] = gpu_indices_val[i];

	num_evals_in_each_thread = num_evals_in_each_thread_val;

	random_seeds = new int[num_threads * num_evals_in_each_thread];

	vsr_array = new double[num_threads * num_evals_in_each_thread];
	discounted_payoff_array = new double[num_threads * num_evals_in_each_thread];

	num_K = num_K_val;
	min_K = min_K_val;
	max_K = max_K_val;
	delta_K	= ( (double) (max_K - min_K) ) / ( (double) (num_K - 1) );

	num_T = num_T_val;
	min_T = min_T_val;
	max_T = max_T_val;
	delta_T = ( (double) (max_T - min_T) ) / ( (double) (num_T - 1) );

	roi_left_T_index = roi_left_T_index_val;
	roi_right_T_index = roi_right_T_index_val;

	vsrs = new Vsr *[num_threads];

	num_q_0_taus = input_num_q_0_taus;

	q_0_taus = new double[num_q_0_taus];
	for (int i = 0; i < num_q_0_taus; i++)
		q_0_taus[i] = input_q_0_taus[i];

	q_0 = new double[num_K * num_q_0_taus];
	for (int i = 0; i < num_K * num_q_0_taus; i++)
		q_0[i] = input_q_0[i];

	c = 0.0;

	average = 0.0;
	std_dev = 0.0;

}

Derivative::~Derivative() {

	delete[] gpu_indices;

	delete[] random_seeds;

	delete[] vsr_array;
	delete[] discounted_payoff_array;

	delete[] vsrs;

	delete[] q_0_taus;
	delete[] q_0;

}

void Derivative::set_params(double const_mu_k_val,
							double const_sigma_k_val,
							double slop_sigma_f_val,
							double corr_coef_val,
							double c_val) {

	const_mu_k = const_mu_k_val;
	const_sigma_k = const_sigma_k_val;
	slop_sigma_f = slop_sigma_f_val;
	corr_coef = corr_coef_val;
	c = c_val;

}

double Derivative::evaluate(double init_vsr) {

	// start of declarations

	int i, j, total_count;

	double *averages_in_threads = new double[num_threads];
	int *count_in_threads = new int[num_threads];

	double *mu_k = new double[num_T * num_T];
	double *sigma_k = new double[num_T];
	double *sigma_f = new double[num_K * num_T];
	double *corr = new double[num_K * num_K];
	double *q_0_new = new double[num_K * num_T];

	// end of declarations

	// mu_k

	for (i = 0; i < num_T; i++)
		for (j = 0; j < num_T; j++)
			mu_k[IDX(i, j, num_T)] =
					this->fun_mu_k(	min_T + i * delta_T,
									min_T + j * delta_T);

//	std::cout << "mu_k" << std::endl;
//	for (i = 0; i < num_T; i++) {
//		for (j = 0; j < num_T; j++)
//			std::cout << mu_k[IDX(i, j, num_T)] << ' ';
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;

	// sigma_k

	for (i = 0; i < num_T; i++)
		sigma_k[i] = this->fun_sigma_k(min_T + i * delta_T);

//	std::cout << "sigma_k" << std::endl;
//	for (i = 0; i < num_T; i++)
//		std::cout << sigma_k[i] << ' ';
//	std::cout << std::endl;
//	std::cout << std::endl;

	// sigma_f

	for (i = 0; i < num_K; i++)
		for (j = 0; j < num_T; j++)
			sigma_f[IDX(i, j, num_K)] =
					this->fun_sigma_f(	min_K + i * delta_K,
										min_T + j * delta_T);

//	std::cout << "sigma_f" << std::endl;
//	for (i = 0; i < num_K; i++) {
//		for (j = 0; j < num_T; j++)
//			std::cout << sigma_f[IDX(i, j, num_K)] << ' ';
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;

	// corr

	for (i = 0; i < num_K; i++)
		for (j = 0; j < num_K; j++)
			corr[IDX(i, j, num_K)] =
					this->fun_corr(	min_K + i * delta_K,
									min_K + j * delta_K);

//	std::cout << "corr" << std::endl;
//	for (i = 0; i < num_K; i++) {
//		for (j = 0; j < num_K; j++)
//			std::cout << corr[IDX(i, j, num_K)] << ' ';
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;

	// q_0

	for (i = 0; i < num_K; i++)
		for (j = 0; j < num_T; j++)
			q_0_new[IDX(i, j, num_K)] =
					this->fun_q_0(	min_K + i * delta_K,
									min_T + j * delta_T);

//	std::cout << "q_0" << std::endl;
//	for (i = 0; i < num_K; i++) {
//		for (j = 0; j < num_T; j++)
//			std::cout << q_0_new[IDX(i, j, num_K)] << ' ';
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;

	for (int i = 0; i < num_threads * num_evals_in_each_thread; i++)
		random_seeds[i] = random() / (RAND_MAX / _MAX_RAND_SEED);

	#ifdef _OMP

		omp_set_num_threads(num_threads);

		#pragma omp parallel default(shared)
		{

			// start of declarations

			int thread_id = omp_get_thread_num();
			int eval_index;

			// end of declarations

			cudaError_t cudaError = cudaSetDevice(gpu_indices[thread_id]);
			if (cudaError != cudaSuccess) {
				printf("GPU selection of %d failed in thread %d.",
						gpu_indices[thread_id],
						thread_id);
			}

			// add new vsr

			vsrs[thread_id] = new Vsr(	num_K,
										min_K,
										max_K,

										num_T,
										min_T,
										max_T,

										roi_left_T_index,
										roi_right_T_index,

										gpu_indices[thread_id]);
			// initialization

			vsrs[thread_id]->init(mu_k, sigma_k, sigma_f, q_0_new, corr, c);

			// simulation

			averages_in_threads[thread_id] = 0.0;
			count_in_threads[thread_id] = 0;

			for (int i = 0; i < num_evals_in_each_thread; i++) {

				eval_index = num_evals_in_each_thread * thread_id + i;

				vsr_array[eval_index] =
						vsrs[thread_id]->sim_vsr(	(init_vsr / 100.0) * (init_vsr / 100.0) * 100.0,
													random_seeds[eval_index]);
				discounted_payoff_array[eval_index] =
						this->discounting() *
						this->payoff(vsr_array[eval_index]);

				if (!std::isnan(discounted_payoff_array[eval_index])) {
					averages_in_threads[thread_id] += discounted_payoff_array[eval_index];
					count_in_threads[thread_id] += 1;
				}

			}

			averages_in_threads[thread_id] =
					averages_in_threads[thread_id] / ((double) count_in_threads[thread_id]);

			// delete temporary variables

			delete vsrs[thread_id];

		}

	#else

		cudaError_t cudaError = cudaSetDevice(0);
		if (cudaError != cudaSuccess) {
			printf("GPU selection of 0 failed in thread 0.");
		}

		// add new vsr

		vsrs[0] = new Vsr(	num_K,
							min_K,
							max_K,

							num_T,
							min_T,
							max_T,

							roi_left_T_index,
							roi_right_T_index,

							0);
		// initialization

		vsrs[thread_id]->init(mu_k, sigma_k, sigma_f, q_0_new, corr, c);

		// simulation

		averages_in_threads[0] = 0.0;
		count_in_threads[0] = 0;

		for (int i = 0; i < num_evals_in_each_thread; i++) {

			discounted_payoff_array[i] =
					this->discounting() *
					this->payoff( vsrs[0]->sim_vsr((init_vsr / 100.0) * (init_vsr / 100.0) * 100.0,
													random_seeds[i]) );

			if (!std::isnan(discounted_payoff_array[i])) {
				averages_in_threads[0] += discounted_payoff_array[i];
				count_in_threads[0] += 1;
			}

		}

		if (count_in_threads[0] != num_evals_in_each_thread)
			printf(	"WARNING: No. 0 thread encounters %d NaNs. \n",
					num_evals_in_each_thread - count_in_threads[0]);

		averages_in_threads[0] =
				averages_in_threads[0] / ((double) count_in_threads[0]);

		// delete temporary variables

		delete vsrs[0];

	#endif

	for (int i = 0; i < num_threads; i++)
		if (count_in_threads[i] != num_evals_in_each_thread)
			printf(	"WARNING: No. %d thread encounters %d NaNs. \n",
					i,
					num_evals_in_each_thread - count_in_threads[i]);

	average = 0.0;
	total_count = 0;
	for (int i = 0; i < num_threads; i++) {
		average += averages_in_threads[i] * count_in_threads[i];
		total_count += count_in_threads[i];
	}
	average = average / ((double) total_count);

	// delete the temporary

	delete[] averages_in_threads;
	delete[] count_in_threads;

	delete[] mu_k;
	delete[] sigma_k;
	delete[] sigma_f;
	delete[] corr;

	delete[] q_0_new;

	return average;

}

double Derivative::get_sd() {

	// start of declarations

	double sum_centered_squares = 0.0;
	int count = 0;

	// end of declarations

	for (int i = 0; i < num_threads * num_evals_in_each_thread; i++)
		if (!std::isnan(discounted_payoff_array[i])) {
			sum_centered_squares +=
					(discounted_payoff_array[i] - average) * (discounted_payoff_array[i] - average);
			count++;
		}

	std_dev = std::sqrt(sum_centered_squares) / ((double) count);

	return std_dev;

}

/*
 * 		VisOpt
 */

double VixOpt::fun_rf_rate(double tau) {

	double rf_rate;

	for (int i = 0; i < num_rf_rates + 1; i++) {
		if (i != num_rf_rates && tau < rf_rates_taus[i]) {
			if (i > 0)
				rf_rate =	( (rf_rates_taus[i] - tau) * rf_rates[i - 1] +
							(tau - rf_rates_taus[i - 1]) * rf_rates[i] ) /
							(rf_rates_taus[i] - rf_rates_taus[i - 1]);
			else
				rf_rate =  	rf_rates[0];
			break;
		}
		if (i == num_rf_rates) {
			rf_rate = rf_rates[num_rf_rates - 1];
		}
		}

	return rf_rate;

}

double VixOpt::fun_mu_k(double t, double tau) {

	double rf_rate, div_rate;

	for (int i = 0; i < num_rf_rates + 1; i++) {
		if (i != num_rf_rates && tau < rf_rates_taus[i]) {
			if (i > 0)
				rf_rate =	( (rf_rates_taus[i] - tau) * rf_rates[i - 1] +
							(tau - rf_rates_taus[i - 1]) * rf_rates[i] ) /
							(rf_rates_taus[i] - rf_rates_taus[i - 1]) +
							slop_mu_k * t;
			else
				rf_rate =  	rf_rates[0] + slop_mu_k * t;
			break;
		}
		if (i == num_rf_rates)
			rf_rate =  	rf_rates[num_rf_rates - 1] +
						slop_mu_k * t;
	}

	for (int i = 0; i < num_div_rates + 1; i++) {
		if (i != num_div_rates && tau < div_rates_taus[i]) {
			if (i > 0)
				div_rate =	( (div_rates_taus[i] - tau) * div_rates[i - 1] +
							(tau - div_rates_taus[i - 1]) * div_rates[i] ) /
							(div_rates_taus[i] - div_rates_taus[i - 1]);
			else
				div_rate = 	div_rates[0];
			break;
		}
		if (i == num_div_rates)
			div_rate = div_rates[num_div_rates - 1];
	}

	return rf_rate - div_rate;

}

double VixOpt::fun_sigma_k(double tau) {

	return intercept_sigma_k + slop_sigma_k * tau;

}

double VixOpt::payoff(double vsr) {

	double temp, result;

	switch (opt_type) {

	case 'c':

		temp = std::sqrt(vsr / 100.0) * 100.0 - strike;
		result =  temp > 0 ? temp : 0.0;

		break;

	case 'p':

		temp = strike - std::sqrt(vsr / 100.0) * 100.0;
		result =  temp > 0 ? temp : 0.0;

		break;

	default:

		printf("Option type %c is not recognized.\n", opt_type);

		break;

	}

//	printf("%p: %4.4f -> %4.4f \n", this, vsr, result);

	return result;

}

double VixOpt::discounting() {

	double result = 1.0;
	double delta_T = ( (double) (max_T - min_T) ) / ( (double) (num_T - 1) );

	for (int i = 1; i < num_T; i++) {
		result *= std::exp( - this->fun_rf_rate(min_T + i * delta_T) * delta_T);
	}

	return result;

}

VixOpt::VixOpt(	int num_K_val,
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
				double const *div_rates_val) :
		Derivative(	num_K_val,
					min_K_val,
					max_K_val,

					num_T_val,
					min_T_val,
					max_T_val,

					roi_left_T_index_val,
					roi_right_T_index_val,

					num_threads_val,
					gpu_indices_val,

					num_evals_in_each_thread_val,

					input_num_q_0_taus,
					input_q_0_taus,
					input_q_0){

	slop_mu_k = 1.0;

	slop_sigma_k = 1.0;
	intercept_sigma_k = 0.0;

	opt_type = 'c';
	strike = 0.0;

	num_rf_rates = num_rf_rates_val;
	rf_rates_taus = new double[num_rf_rates];
	rf_rates = new double[num_rf_rates];

	num_div_rates = num_div_rates_val;
	div_rates_taus = new double[num_div_rates];
	div_rates = new double[num_div_rates];

	for (int i = 0; i < num_rf_rates; i++) {
		rf_rates_taus[i] = rf_rates_taus_val[i];
		rf_rates[i] = rf_rates_val[i];
	}

	for (int i = 0; i < num_div_rates; i++) {
		div_rates_taus[i] = div_rates_taus_val[i];
		div_rates[i] = div_rates_val[i];
	}

}

VixOpt::~VixOpt() {

	delete[] rf_rates_taus;
	delete[] rf_rates;

	delete[] div_rates_taus;
	delete[] div_rates;

}

void VixOpt::simulate(	double slop_mu_k_val,
						double slop_sigma_k_val,
						double intercept_sigma_k_val,
						double slop_sigma_f_val,
						double corr_coef_val,

						double c_val,
						double init_vsr,

						bool save_to_file) {

	slop_mu_k = slop_mu_k_val;
	slop_sigma_k = slop_sigma_k_val;
	intercept_sigma_k = intercept_sigma_k_val;
	slop_sigma_f = slop_sigma_f_val;
	corr_coef = corr_coef_val;
	c = c_val;

	// start of declarations

	int i, j;

	double *mu_k = new double[num_T * num_T];
	double *sigma_k = new double[num_T];
	double *sigma_f = new double[num_K * num_T];
	double *corr = new double[num_K * num_K];
	double *q_0_new = new double[num_K * num_T];

	// end of declarations

	// mu_k

	for (i = 0; i < num_T; i++)
		for (j = 0; j < num_T; j++)
			mu_k[IDX(i, j, num_T)] =
					this->fun_mu_k(	min_T + i * delta_T,
									min_T + j * delta_T);

//	std::cout << "mu_k" << std::endl;
//	for (i = 0; i < num_T; i++) {
//		for (j = 0; j < num_T; j++)
//			std::cout << mu_k[IDX(i, j, num_T)] << ' ';
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;

	// sigma_k

	for (i = 0; i < num_T; i++)
		sigma_k[i] = this->fun_sigma_k(min_T + i * delta_T);

//	std::cout << "sigma_k" << std::endl;
//	for (i = 0; i < num_T; i++)
//		std::cout << sigma_k[i] << ' ';
//	std::cout << std::endl;
//	std::cout << std::endl;

	// sigma_f

	for (i = 0; i < num_K; i++)
		for (j = 0; j < num_T; j++)
			sigma_f[IDX(i, j, num_K)] =
					this->fun_sigma_f(	min_K + i * delta_K,
										min_T + j * delta_T);

//	std::cout << "sigma_f" << std::endl;
//	for (i = 0; i < num_K; i++) {
//		for (j = 0; j < num_T; j++)
//			std::cout << sigma_f[IDX(i, j, num_K)] << ' ';
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;

	// corr

	for (i = 0; i < num_K; i++)
		for (j = 0; j < num_K; j++)
			corr[IDX(i, j, num_K)] =
					this->fun_corr(	min_K + i * delta_K,
									min_K + j * delta_K);

//	std::cout << "corr" << std::endl;
//	for (i = 0; i < num_K; i++) {
//		for (j = 0; j < num_K; j++)
//			std::cout << corr[IDX(i, j, num_K)] << ' ';
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;

	// q_0

	for (i = 0; i < num_K; i++)
		for (j = 0; j < num_T; j++)
			q_0_new[IDX(i, j, num_K)] =
					this->fun_q_0(	min_K + i * delta_K,
									min_T + j * delta_T);

//	std::cout << "q_0" << std::endl;
//	for (i = 0; i < num_K; i++) {
//		for (j = 0; j < num_T; j++)
//			std::cout << q_0_new[IDX(i, j, num_K)] << ' ';
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;

	for (int i = 0; i < num_threads * num_evals_in_each_thread; i++)
		random_seeds[i] = random() / (RAND_MAX / _MAX_RAND_SEED);

	#ifdef _OMP

		omp_set_num_threads(num_threads);

		#pragma omp parallel default(shared)
		{

			// start of declarations

			int thread_id = omp_get_thread_num();
			int eval_index;

			// end of declarations

			cudaError_t cudaError = cudaSetDevice(gpu_indices[thread_id]);
			if (cudaError != cudaSuccess) {
				printf("GPU selection of %d failed in thread %d.",
						gpu_indices[thread_id],
						thread_id);
			}

			// add new vsr

			vsrs[thread_id] = new Vsr(	num_K,
										min_K,
										max_K,

										num_T,
										min_T,
										max_T,

										roi_left_T_index,
										roi_right_T_index,

										gpu_indices[thread_id]);
			// initialization

			vsrs[thread_id]->init(mu_k, sigma_k, sigma_f, q_0_new, corr, c);

			// simulation

			for (int i = 0; i < num_evals_in_each_thread; i++) {

				eval_index = num_evals_in_each_thread * thread_id + i;

				vsr_array[eval_index] =
						vsrs[thread_id]->sim_vsr(	(init_vsr / 100.0) * (init_vsr / 100.0) * 100.0,
													random_seeds[eval_index],
													save_to_file);

			}

			// delete temporary variables

			delete vsrs[thread_id];

		}

	#else

		cudaError_t cudaError = cudaSetDevice(0);
		if (cudaError != cudaSuccess) {
			printf("GPU selection of 0 failed in thread 0.");
		}

		// add new vsr

		vsrs[0] = new Vsr(	num_K,
							min_K,
							max_K,

							num_T,
							min_T,
							max_T,

							roi_left_T_index,
							roi_right_T_index,

							0);
		// initialization

		vsrs[thread_id]->init(mu_k, sigma_k, sigma_f, q_0_new, corr, c);

		// simulation

		for (int i = 0; i < num_evals_in_each_thread; i++) {

			vsr_array[i] = vsrs[0]->sim_vsr((init_vsr / 100.0) * (init_vsr / 100.0) * 100.0,
											random_seeds[i],
											save_to_file);

		}

		// delete temporary variables

		delete vsrs[0];

	#endif

	// delete the temporary

	delete[] mu_k;
	delete[] sigma_k;
	delete[] sigma_f;
	delete[] corr;

	delete[] q_0_new;

}

double VixOpt::evaluate(char opt_type_val, double strike_val) {

	opt_type = opt_type_val;
	strike = strike_val;

	// start of declarations

	int total_count;

	double *sums_in_threads = new double[num_threads];
	int *count_in_threads = new int[num_threads];

	// end of declarations

	#ifdef _OMP

		omp_set_num_threads(num_threads);

		#pragma omp parallel default(shared)
		{

			// start of declarations

			int thread_id = omp_get_thread_num();
			int eval_index;

			// end of declarations

			sums_in_threads[thread_id] = 0.0;
			count_in_threads[thread_id] = 0;

			for (int i = 0; i < num_evals_in_each_thread; i++) {

				eval_index = num_evals_in_each_thread * thread_id + i;

				discounted_payoff_array[eval_index] =
						this->discounting() *
						this->payoff(vsr_array[eval_index]);

				if (!std::isnan(discounted_payoff_array[eval_index])) {
					sums_in_threads[thread_id] += discounted_payoff_array[eval_index];
					count_in_threads[thread_id] += 1;
				}

			}

		}

	#else

		sums_in_threads[0] = 0.0;
		count_in_threads[0] = 0;

		for (int i = 0; i < num_evals_in_each_thread; i++) {

			discounted_payoff_array[i] =
					this->discounting() *
					this->payoff(vsr_array[eval_index]);

			if (!std::isnan(discounted_payoff_array[i])) {
				sums_in_threads[0] += discounted_payoff_array[i];
				count_in_threads[0] += 1;
			}

		}

		if (count_in_threads[0] != num_evals_in_each_thread)
			printf(	"WARNING: No. 0 thread encounters %d NaNs. \n",
					num_evals_in_each_thread - count_in_threads[0]);

	#endif

	for (int i = 0; i < num_threads; i++)
		if (count_in_threads[i] != num_evals_in_each_thread)
			printf(	"WARNING: No. %d thread encounters %d NaNs. \n",
					i,
					num_evals_in_each_thread - count_in_threads[i]);

	average = 0.0;
	total_count = 0;
	for (int i = 0; i < num_threads; i++) {
		average += sums_in_threads[i];
		total_count += count_in_threads[i];
	}
	average = average / total_count;

	// delete the temporary

	delete[] sums_in_threads;
	delete[] count_in_threads;

	return average;


}
