//
// Created by xixuan on 4/3/17.
//

#ifndef MUL_NOR_SAMPLING_H
#define MUL_NOR_SAMPLING_H

#define IDX(i, j, ld_i) (((j)*(ld_i))+(i))

void norm_transform_mat(double *result,
                        const double *covar_array,
                        int dim);

void draw_from_mul_gaussian(double *result,                  		// pointer to the result
                            int size,                       		// Dimensionality (rows)
                            int nn,                         		// How many samples (columns) to draw
                            const double *normTransform_array,		// transform mat
                            int random_seed = 1,            		// set random seed
                            bool show_result = false        		// show result when running
);

#endif //MUL_NOR_SAMPLING_H
