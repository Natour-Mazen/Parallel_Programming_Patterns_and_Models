#pragma once

#include <StudentWork.h>
#include <vector>
#include <functional>
#include <utility>
#include <numeric>
#include "inclusive_scan.h"

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
		std::function<T(T,T)>& functor
	) {
        // std::inclusive_scan(input.begin(),input.end(),output.begin(),functor);
        // OR
        if (input.empty()) {
          return;
        }

        output[0] = input[0];

        for(int i=1; i < input.size(); ++i){
          output[i] = functor(output[i-1],input[i]);
        }
	}
	
	template< typename T>
	void run_scan_parallel(
		std::vector<T>& input,
		std::vector<T>& output,
		std::function<T(T,T)>& functor
	) {		
		OPP::inclusive_scan(
			input.begin(), 
			input.end(), 
			output.begin(), 
			std::forward<std::function<T(T,T)>>(functor)
		);
	}
};
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/