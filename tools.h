//
// Created by xixuan on 10/10/16.
//

#ifndef TOOLS_H
#define TOOLS_H

#include <ctime>
#include <string>

#define IDX(i, j, ld_i) (((j)*(ld_i))+(i))
#define IDXX(i, j, k, ld_i, ld_j) (((k)*((ld_i)*(ld_j)))+((j)*(ld_i))+(i))

// elapsed time measurer
class Time_measurer {
private:
	std::time_t begin_time;
	std::time_t end_time;
public:
	Time_measurer();

	~Time_measurer();
};

void print_to_file(std::string const &file_name, double const *mat_ptr, int num_rows, int num_cols);

#endif
