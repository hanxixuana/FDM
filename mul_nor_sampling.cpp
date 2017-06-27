//
// Created by xixuan on 4/3/17.
//

#include "mul_nor_sampling.h"

#include "Dense"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

/*
 * FROM: http://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
 *
 * Note:	1. The original version uses NullaryExpr and a constructed functor. But it does not seem to be ok
 * 				with multiple threads.
 * 			2. So I changed it to a much obsolete way.
 */

/*
  We need a functor that can pretend it's const,
  but to be a good random number generator
  it needs mutable state.
*/

typedef Eigen::Matrix< long double, Eigen::Dynamic, Eigen::Dynamic > Matrix;
typedef Eigen::Matrix< long double, Eigen::Dynamic, 1> Vector;

void norm_transform_mat(double *result,
                        const double *covar_array,
                        int dim) {

    Matrix covar = Eigen::Map<Eigen::MatrixXd>(const_cast<double*>(covar_array), dim, dim).cast<long double>();

    Matrix normTransform(dim, dim);

    Eigen::LLT<Matrix> cholSolver(covar);

    if (cholSolver.info() == Eigen::Success) {
        // Use cholesky solver
        normTransform = cholSolver.matrixL();

    } else {
        // Use eigen solver
        Eigen::SelfAdjointEigenSolver<Matrix> eigenSolver(covar);
        normTransform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::Map<Eigen::MatrixXd>(result, dim, dim) = normTransform.cast<double>();

}

void draw_from_mul_gaussian(double *result,                 // pointer to the result
                            int size,                       // Dimensionality (rows)
                            int nn,                         // How many samples (columns) to draw
                            const double *normTransform_array,      // transform matrix
                            int random_seed,                // set random seed
                            bool show_result                // show result when running
) {
    /*
     * Draw nn samples from a size-dimensional normal distribution with a specified mean and covariance
     */

    if (nn > 1) {

        boost::mt19937 rng;    // The uniform pseudo-random algorithm
        rng.seed((const uint32_t)random_seed);

        boost::normal_distribution<long double> norm;  // The gaussian combinator

        Matrix normTransform =
                Eigen::Map<Eigen::MatrixXd>(const_cast<double*>(normTransform_array),
                                            size,
                                            size).cast<long double>();

        Matrix std_gaussian(size, nn);

        for (int j = 0; j < nn; j++)
            for (int i = 0; i < size; i++)
                std_gaussian(i, j) = norm(rng);

        Matrix samples = normTransform * std_gaussian;

        Eigen::Map<Eigen::MatrixXd>(result, samples.rows(), samples.cols()) = samples.cast<double>();

        if (show_result) {
            std::cout << "Samples" << std::endl;

            for (int i = 0; i < size; i++) {
                for(int j = 0; j < nn; j++)
                    std::cout << result[IDX(i, j, size)] << ' ';
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    else {
        for (int i = 0; i < size; i++)
            result[i] = 0.0;
    }

}

