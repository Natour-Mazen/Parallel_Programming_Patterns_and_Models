#pragma once
#include <execution>
#include <algorithm>
#include <StudentWork.h>
#include <vector>

class StudentWorkImpl: public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWorkImpl() = default; 
	StudentWorkImpl(const StudentWorkImpl&) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

	template<typename T>
	T run_sum(const std::vector<T>& input) 
	{
		// The parallel sum
		return std::reduce(std::execution::par_unseq, input.begin(), input.end(), T(0));
	}
	
	template<typename T>
	T run_sum_square(const std::vector<T>& input)
	{
		// The parallel square and then sum
        std::vector<T> sqr(input.size());
        std::transform(std::execution::par_unseq, input.begin(), input.end(),
                       sqr.begin(), [](T a){ return a*a;});

        return run_sum(sqr);
	}

	template<typename T>
	T run_sum_square_opt(const std::vector<T>& input)
	{
		// The parallel square and sum in once
        return std::transform_reduce(std::execution::par_unseq, input.begin(), input.end(), input.begin(),
                                     T(0), std::plus<T>{}, std::multiplies<T>{});
	}
};
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/