#include "optimization.h"
#include "evaluation.h"
#include "tools.h"
#include "mat.h"

#include <iostream>

#define IDX(i, j, ld_i) (((j)*(ld_i))+(i))


int main() {

	Time_measurer tm;

	// start of making the initial surface

	const int num_q_0_strikes = 13, num_q_0_taus = 8;
	double min_q_0_strike = 2250, max_q_0_strike = 2550;

	// 3-May-2017 from Bloomberg
	double q_0_taus[num_q_0_taus] = {
			0.0, 0.04, 0.12, 0.22, 0.30, 0.38, 0.63, 0.73
	};
	double spx_calls[num_q_0_strikes * num_q_0_taus] = {
			// 3-May-2017 current price 2388.13
			138.13, 113.13, 88.13, 63.13, 38.13, 13.13, 0.00,
			0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
			// 19-May-17 (16d); CSize 100; IDiv 3.52; R .96; FF 2390.12
			140.25, 115.80, 91.65, 68.00, 45.30, 24.90, 10.15,
			2.58, 0.43, 0.17, 0.17, 0.15, 0.13,
			// 16-Jun-17 (44d); CSize 100; IDiv 2.55; R 1.01; FF 2390.12
			143.70, 120.45, 97.80, 76.10, 55.80, 37.55, 22.60,
			11.90, 5.35, 2.27, 1.05, 0.60, 0.43,
			// 21-Jul-17 (79d); CSize 100; IDiv 1.91; R 1.12; FF 2390.12
			152.10, 130.00, 108.70, 88.35, 69.15, 51.95, 36.70, 24.15,
			14.70, 8.25, 4.50, 2.38, 1.30,
			// 18-Aug-17 (107d); CSize 100; IDiv 2.00; R 1.22; FF 2390.12
			157.90, 136.70, 116.20, 96.50, 78.45, 61.35, 46.30, 33.25,
			22.60, 14.45, 8.80, 5.25, 3.10,
			// 15-Sep-17 (135d); CSize 100; IDiv 1.98; R 1.30; FF 2390.12
			164.10, 143.50, 123.50, 104.65, 86.65, 70.10, 54.85, 41.60,
			30.30, 21.05, 14.10, 9.05, 5.75,
			// 15-Dec-17 (226d); CSize 100; IDiv 1.88; R 1.51; FF 2390.12
			184.50, 165.45, 146.85, 129.30, 112.50, 96.55, 81.70, 67.90,
			55.50, 44.50, 34.85, 26.80, 20.20,
			// 19-Jan-18 (261d); CSize 100; IDiv 1.78; R 1.58; FF 2390.12
			192.75, 174.05, 155.75, 138.35, 121.65, 105.75, 90.85, 77.05,
			64.40, 52.90, 42.75, 33.90, 26.50
	};

	double *dev_spx_calls;
	dev_alloc_and_init(spx_calls, dev_spx_calls, "dev_spx_calls", num_q_0_strikes * num_q_0_taus);
	double *dev_1st_dev_spx_calls;
	dev_alloc(dev_1st_dev_spx_calls, "dev_1st_dev_spx_calls", num_q_0_strikes * num_q_0_taus);
	double *dev_q_0;
	dev_alloc(dev_q_0, "dev_q_0", num_q_0_strikes * num_q_0_taus);

	col_1st_pd(dev_spx_calls, dev_1st_dev_spx_calls, num_q_0_strikes, num_q_0_taus, 1.0);
	col_1st_pd(dev_1st_dev_spx_calls, dev_q_0, num_q_0_strikes, num_q_0_taus, 1.0);
	colwise_normalization(dev_q_0, num_q_0_strikes, num_q_0_taus);
	set_vec_to_val(dev_q_0, num_q_0_strikes, 0.0);
	set_vec_to_val(dev_q_0 + 6, 1, 1.0);

	double *q_0 = new double[num_q_0_strikes * num_q_0_taus];
	dev_download(q_0, dev_q_0, "dev_q_0", num_q_0_strikes * num_q_0_taus);

	if (false) print(q_0, num_q_0_strikes, num_q_0_taus);

	// end of making the initial surface

	// start of making risk-free rates and dividend rates

	const int num_rates_taus = 13;
	double rates_taus[num_rates_taus] = {
			0.00, 0.04, 0.06, 0.10, 0.14, 0.22, 0.29,
			0.37, 0.41, 0.50, 0.63, 0.66, 0.72
	};
	double rf_rates[num_rates_taus] = {
			0.0117, 0.0117, 0.0117, 0.0117, 0.0117, 0.0117, 0.0121,
			0.0127, 0.0129, 0.0131, 0.0133, 0.0134, 0.0135
	};
	double div_rates[num_rates_taus] = {
			0.0281, 0.0348, 0.0302, 0.0256, 0.0222, 0.0190, 0.0199,
			0.0197, 0.0187, 0.0172, 0.0187, 0.0182, 0.0176
	};

	// end of making risk-free rates and dividend rates

	// start of making vix call prices

	const int num_vix_calls_strikes = 5;
	const int num_vix_calls_maturities = 2;
	double vix_calls_strikes[num_vix_calls_strikes * num_vix_calls_maturities] = {
			// 17-May-17 (14d); CSize 100; R .96; UXK7 12.10
//			9.5, 10, 10.5, 11, 11.5,
			// 21-Jun-17 (49d); CSize 100; R 1.02; UXM7 13.00
			9.5, 10, 10.5, 11, 11.5,
			// 19-Jul-17 (77d); CSize 100; R 1.11; UXN7 13.90
			10, 11, 12, 13, 14,
			// 16-Aug-17 (105d); CSize 100; R 1.21; UXQ7 14.38
//			10, 11, 12, 13, 14
	};
	double vix_calls[num_vix_calls_strikes * num_vix_calls_maturities] = {
			// 17-May-17 (14d); CSize 100; R .96; UXK7 12.10
//			2.58, 2.08, 1.60, 1.18, 0.90,
			// 21-Jun-17 (49d); CSize 100; R 1.02; UXM7 13.00
			3.50, 2.95, 2.52, 2.15, 1.88,
			// 19-Jul-17 (77d); CSize 100; R 1.11; UXN7 13.90
			3.90, 3.10, 2.43, 2.00, 1.70,
			// 16-Aug-17 (105d); CSize 100; R 1.21; UXQ7 14.38
//			4.35, 3.55, 2.92, 2.48, 2.10
	};

//	int num_T_val[num_vix_calls_maturities] = {7, 13, 16, 20};
//	double min_T_val[num_vix_calls_maturities] = {0, 0, 0, 0};
//	double max_T_val[num_vix_calls_maturities] = {0.1167, 0.2139, 0.2917, 0.3694};
//
//	int roi_left_T_index_val[num_vix_calls_maturities] = {2, 7, 11, 15};
//	int roi_right_T_index_val[num_vix_calls_maturities] = {6, 12, 15, 19};

	int num_T_val[num_vix_calls_maturities] = {13, 16};
	double min_T_val[num_vix_calls_maturities] = {0, 0};
	double max_T_val[num_vix_calls_maturities] = {0.2139, 0.2917};

	int roi_left_T_index_val[num_vix_calls_maturities] = {7, 11};
	int roi_right_T_index_val[num_vix_calls_maturities] = {12, 15};

	// end of making vix call prices

	// start of making forward variance swap rates

	const int num_forward_vsrs_taus = 9;
	double forward_vsrs_taus[num_forward_vsrs_taus] = {
			0.09, 0.17, 0.26, 0.34, 0.43, 0.51, 0.59, 0.68, 0.77
	};
	double forward_vsrs[num_forward_vsrs_taus] = {
			13.96, 14.21, 15.85, 16.73, 16.59, 17.21, 17.66, 16.97, 18.86
	};

	// end of making forward variance swap rates

	// start of selecting gpus and declare results

	const int num_threads = 6, num_evals_in_each_thread = 500;
	int gpu_choose[num_threads] = {0, 0, 0, 1, 1, 1};

	// end of selecting gpus and declare results

	// optimize parameters

//	No.	slop_mu_k	slop_sigma_k	int_sigma_k	slop_sigma_f	corr_coef	MSPE
//	0	0.007500	0.041000	0.001400	0.150000	-0.010000	0.005758
//	1	0.007500	0.041000	0.001400	0.100000	-0.010000	0.006164
//	2	0.007500	0.041000	0.001400	0.050000	-0.010000	0.006533

//	No.	slop_mu_k	slop_sigma_k	int_sigma_k	slop_sigma_f	corr_coef	MSPE
//	0	0.007500	0.041000	0.001400	0.225000	-0.010000	0.005422
//	1	0.007500	0.041000	0.001400	0.225000	-0.012500	0.005446
//	2	0.007500	0.041000	0.001400	0.225000	-0.015000	0.005541
//	3	0.007500	0.041000	0.001400	0.187500	-0.015000	0.005573

//	No.	slop_mu_k	slop_sigma_k	int_sigma_k	slop_sigma_f	corr_coef	MSPE
//	0	0.007500	0.041000	0.000980	0.157500	-0.010000
//	1	0.007500	0.041000	0.000980	0.225000	-0.010000	0.004753
//	2	0.007500	0.028700	0.001820	0.292500	-0.010000	0.004789
//	3	0.007500	0.041000	0.000980	0.191250	-0.010000	0.004805
//	4	0.009750	0.041000	0.001400	0.191250	-0.010000	0.005188
//	5	0.009750	0.041000	0.001400	0.225000	-0.010000	0.005193

//	No.	slop_mu_k	slop_sigma_k	int_sigma_k	slop_sigma_f	corr_coef	MSPE
//	0	0.007500	0.041000	0.000980	0.110250	-0.013000	0.003934
//	1	0.007500	0.041000	0.000980	0.133875	-0.011500	0.003971
//	2	0.007500	0.041000	0.000980	0.133875	-0.010000	0.003997
//	3	0.007500	0.041000	0.000980	0.133875	-0.013000	0.004012
//	4	0.007500	0.041000	0.000980	0.133875	-0.008500	0.004015
//	5	0.007500	0.041000	0.000980	0.157500	-0.013000	0.004067

//	No.	slop_mu_k	slop_sigma_k	int_sigma_k	slop_sigma_f	corr_coef	MSPE
//	0	0.007500	0.041000	0.000980	0.110250	-0.016900	0.003835
//	1	0.007500	0.041000	0.000980	0.110250	-0.013000	0.003934
//	2	0.007500	0.041000	0.000980	0.143325	-0.009100	0.003998
//	3	0.007500	0.041000	0.000980	0.143325	-0.013000	0.004196
//	4	0.007500	0.041000	0.000980	0.077175	-0.009100	0.004279
//	5	0.005250	0.041000	0.001274	0.077175	-0.009100	0.004283

//	No.	slop_mu_k	slop_sigma_k	int_sigma_k	slop_sigma_f	corr_coef	MSPE
//	0	0.007500	0.041000	0.000980	0.102900	-0.019153	0.003818
//	1	0.007500	0.041000	0.000980	0.102900	-0.018027	0.003824
//	2	0.007500	0.041000	0.000980	0.102900	-0.020280	0.003826
//	3	0.007500	0.041000	0.000980	0.110250	-0.016900	0.003835
//	4	0.007500	0.041000	0.000980	0.110250	-0.018027	0.003835

//	No.	slop_mu_k	slop_sigma_k	int_sigma_k	slop_sigma_f	corr_coef	MSPE
//	0	0.009750	0.041000	0.000833	0.102900	-0.013407	0.003540
//	1	0.009750	0.041000	0.000833	0.102900	-0.015322	0.003705
//	2	0.009750	0.041000	0.000833	0.102900	-0.021068	0.003811
//	3	0.007500	0.041000	0.000980	0.102900	-0.019153	0.003818
//	4	0.007500	0.041000	0.000980	0.102900	-0.021068	0.003838

	if (false) {

		double radius = 0.3;

		double start_slop_mu_k = 0.009750;
		int try_num_slop_mu_k = 0;

		double start_slop_sigma_k = 0.041000;
		int try_num_slop_sigma_k = 0;

		double start_intercept_sigma_k = 0.000833;
		int try_num_intercept_sigma_k = 0;

		double start_slop_sigma_f = 0.102900;
		int try_num_slop_sigma_f = 0;

		double start_corr_coef = -0.013407;
		int try_num_corr_coef = 0;

		Optim optim(	num_q_0_strikes, min_q_0_strike, max_q_0_strike,
						num_T_val, min_T_val, max_T_val,
						roi_left_T_index_val, roi_right_T_index_val,

						num_threads, gpu_choose, num_evals_in_each_thread,

						num_q_0_taus, q_0_taus, q_0,

						num_rates_taus, rates_taus, rf_rates,
						num_rates_taus, rates_taus, div_rates,

						try_num_slop_mu_k, try_num_slop_sigma_k, try_num_intercept_sigma_k,
						try_num_slop_sigma_f, try_num_corr_coef,

						num_vix_calls_strikes, num_vix_calls_maturities, vix_calls_strikes, vix_calls,

						num_forward_vsrs_taus, forward_vsrs_taus, forward_vsrs);

		optim.set_param_candidates(	start_slop_mu_k, radius * std::abs(start_slop_mu_k),
									start_slop_sigma_k, radius * std::abs(start_slop_sigma_k),
									start_intercept_sigma_k, radius * std::abs(start_intercept_sigma_k),
									start_slop_sigma_f, radius * std::abs(start_slop_sigma_f),
									start_corr_coef, radius * std::abs(start_corr_coef));

		optim.optimize('c', 0.0, 0);

		printf("\n\n ===== SUMMARY ===== \n");
		printf("No.\tslop_mu_k\tslop_sigma_k\tint_sigma_k\tslop_sigma_f\tcorr_coef\tMSPE\n");
		for (	int i = 0;
				i < (2 * try_num_slop_mu_k + 1) *
					(2 * try_num_slop_sigma_k + 1) *
					(2 * try_num_intercept_sigma_k + 1) *
					(2 * try_num_slop_sigma_f + 1) *
					(2 * try_num_corr_coef + 1);
				i++) {
			printf(	"%d\t%f\t%f\t%f\t%f\t%f\t%f\t\n",
					i,
					optim.get_sorted_mspe(i)->slop_mu_k,
					optim.get_sorted_mspe(i)->slop_sigma_k,
					optim.get_sorted_mspe(i)->intercept_sigma_k,
					optim.get_sorted_mspe(i)->slop_sigma_f,
					optim.get_sorted_mspe(i)->corr_coef,
					optim.get_sorted_mspe(i)->mspe);
		}
		printf("\n\n");

	}

	// end of optimizing parameters

	// make data for the video

	if (false) {

		srand(0);

		VixOpt option(	num_q_0_strikes, min_q_0_strike, max_q_0_strike,
						20, 0.0, 0.3694,
						15, 19,

						1, gpu_choose, 30,

						num_q_0_taus, q_0_taus, q_0,
						num_rates_taus, rates_taus, rf_rates,
						num_rates_taus, rates_taus, div_rates);

		option.simulate(0.0075, 0.041, 0.00098, 0.110250, -0.013,
						0.0, 0.0, true);

		}

	// end of making data for the video

	// price all vix calls

	if (true) {

		const int num_all_vix_calls_maturities = 4;

		double vix_all_calls_strikes[num_vix_calls_strikes * num_all_vix_calls_maturities] = {
				// 17-May-17 (14d); CSize 100; R .96; UXK7 12.10
				9.5, 10, 10.5, 11, 11.5,
				// 21-Jun-17 (49d); CSize 100; R 1.02; UXM7 13.00
				9.5, 10, 10.5, 11, 11.5,
				// 19-Jul-17 (77d); CSize 100; R 1.11; UXN7 13.90
				10, 11, 12, 13, 14,
				// 16-Aug-17 (105d); CSize 100; R 1.21; UXQ7 14.38
				10, 11, 12, 13, 14
		};

		int all_num_T_val[num_all_vix_calls_maturities] = {7, 13, 16, 20};
		double all_min_T_val[num_all_vix_calls_maturities] = {0, 0, 0, 0};
		double all_max_T_val[num_all_vix_calls_maturities] = {0.1167, 0.2139, 0.2917, 0.3694};

		int all_roi_left_T_index_val[num_all_vix_calls_maturities] = {2, 7, 11, 15};
		int all_roi_right_T_index_val[num_all_vix_calls_maturities] = {6, 12, 15, 19};

		double all_init_vsrs[num_all_vix_calls_maturities] = {13.96, 14.068672, 15.010199, 16.197947};

		double results[num_vix_calls_strikes * num_all_vix_calls_maturities];
		double sds[num_vix_calls_strikes * num_all_vix_calls_maturities];

		for (int i = 0; i < num_all_vix_calls_maturities; i++) {
			srand(0);

			VixOpt option(	num_q_0_strikes, min_q_0_strike, max_q_0_strike,
							all_num_T_val[i], all_min_T_val[i], all_max_T_val[i],
							all_roi_left_T_index_val[i], all_roi_right_T_index_val[i],

							num_threads, gpu_choose, num_evals_in_each_thread * 2,

							num_q_0_taus, q_0_taus, q_0,
							num_rates_taus, rates_taus, rf_rates,
							num_rates_taus, rates_taus, div_rates);

			option.simulate(0.009750, 0.041000, 0.000833, 0.102900, -0.013407,
							0.0, all_init_vsrs[i]);

			for (int j = 0; j < num_vix_calls_strikes; j++) {
				results[IDX(j, i, num_vix_calls_strikes)] =
						option.evaluate('c',
										vix_all_calls_strikes[IDX(j, i, num_vix_calls_strikes)]);
				sds[IDX(j, i, num_vix_calls_strikes)] = option.get_sd();
			}

		}

		print_to_file(	"./output/prices.csv",
						results,
						num_vix_calls_strikes,
						num_all_vix_calls_maturities);

		print_to_file(	"./output/sds.csv",
						sds,
						num_vix_calls_strikes,
						num_all_vix_calls_maturities);

	}

	// end of pricing all vix calls

	// release memory

	dev_release(dev_spx_calls, "dev_spx_calls");
	dev_release(dev_1st_dev_spx_calls, "dev_1st_dev_spx_calls");
	dev_release(dev_q_0, "dev_q_0");

	delete[] q_0;

	return EXIT_SUCCESS;
}
