#include <execution>
#include <algorithm>
#include <iostream>
#include <exo1/student.h>

namespace {
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_square(const std::vector<int>& input, std::vector<int>& output) 
{
	// The std::transform (aka MAP) pattern in parallel mode
	std::transform(std::execution::par_unseq, input.begin(), input.end(), input.begin(),
				   output.begin(), std::multiplies<int>{});
    // OR
    //std::transform(std::execution::par_unseq, input.begin(), input.end(), output.begin(),
     //              [](int a) { return a * a; });
}

void StudentWorkImpl::run_sum(
	const std::vector<int>& input_a,
	const std::vector<int>& input_b,
	std::vector<int>& output
) {
    // The parallel sum using std::transform
    std::transform(std::execution::par_unseq, input_a.begin(), input_a.end(), input_b.begin(),
    			   output.begin(), std::plus<int>{});
    //std::transform(std::execution::par_unseq, input_a.begin(), input_a.end(), input_b.begin(),
    //               output.begin(), [](int a, int b) { return a + b; });
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/