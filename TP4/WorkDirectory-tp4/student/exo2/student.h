#pragma once

#include <StudentWork.h>
#include <vector>
#include <functional>
#include "exclusive_scan.h"
#include <numeric>

class StudentWorkImpl: public StudentWork
{
public:

	bool isImplemented() const {
		return true;
	}

	StudentWorkImpl() = default; 
	StudentWorkImpl(const StudentWorkImpl&) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

	template< typename T>
	void run_scan_sequential(
		std::vector<T>& input,
		std::vector<T>& output,
		const T& Tinit,
		std::function<T(T,T)>& functor
	) {
    //std::exclusive_scan(input.begin(), input.end(), output.begin(), Tinit, functor);
    // OR
   if (input.empty()) {
      return;
    }

    output.resize(input.size());

    output[0] = Tinit;

    for (size_t i = 1; i < input.size(); ++i) {
      output[i] = functor(output[i - 1], input[i - 1]);
    }
	}
	
	template< typename T>
	void run_scan_parallel(
		std::vector<T>& input,
		std::vector<T>& output,
		const T& Tinit,
		std::function<T(T,T)>& functor
	) {		
		OPP::exclusive_scan(
			input.begin(), 
			input.end(), 
			output.begin(), 
			std::forward<std::function<T(T,T)>>(functor),
			Tinit
		);
	}
};
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/