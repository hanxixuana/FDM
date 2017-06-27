//
// Created by xixuan on 10/10/16.
//

#include "tools.h"

#include <iostream>
#include <chrono>
#include <fstream>

Time_measurer::Time_measurer() {
	begin_time = std::clock();
	end_time = begin_time;
	std::cout << std::endl
			  << "Timer at " << this
			  << " ---> Start recording time..."
			  << std::endl << std::endl;
}

Time_measurer::~Time_measurer() {
	end_time = std::clock();
	std::cout << std::endl
			  << "Timer at " << this
			  << " ---> " << "Elapsed Time: "
			  << double(end_time - begin_time) / CLOCKS_PER_SEC
			  << " seconds"
			  << std::endl << std::endl;
}

void print_to_file(std::string const &file_name, double const *mat_ptr, int num_rows, int num_cols) {

    std::ofstream file;
    file.open(file_name);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            file << mat_ptr[IDX(i, j, num_rows)] << '\t';
        }
        file << std::endl;
    }
    file << std::endl;

    file.close();

}
