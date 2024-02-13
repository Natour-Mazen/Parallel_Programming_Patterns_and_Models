#include <iostream>
#include <exo3/student.h>

#include <exo3/transform.h>

namespace {
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_square(const std::vector<int>& input, std::vector<int>& output) 
{
	// The OPP:transform (aka MAP) pattern in parallel mode
    OPP::transform(input.begin(),
                   input.end(),
                   output.begin(),
                   [] (const int& a){ return a * a; }
                   );
}

void StudentWorkImpl::run_sum(
	const std::vector<int>& input_a,
	const std::vector<int>& input_b,
	std::vector<int>& output
) {
	// parallel sum using OPP:transform
    // The OPP:transform (aka MAP) pattern in parallel mode
    OPP::transform(input_a.begin(),
                   input_a.end(),
                   input_b.begin(),
                   output.begin(),
                   [] (const int& a, const int& b){ return a + b; }
    );
	
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/