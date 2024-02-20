#pragma once

#include <StudentWork.h>
#include <vector>
#include <functional>
#include <exo4/partition.h>
#include <previous/transform.h>



class StudentWorkImpl: public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWorkImpl() = default; 
	StudentWorkImpl(const StudentWorkImpl&) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

	template<typename T>
	void run_radixSort_parallel(
		std::vector<T>& input,
		std::vector<T>& output
	) {
		std::copy(input.begin(), input.end(), output.begin());
		std::vector<T> temp(input.size());
		std::vector<T>* array[2] = { &output, &temp }; // des pointeurs conviennent aussi !
		for(unsigned numeroBit=0; numeroBit<sizeof(T)*8; ++numeroBit) 
		{
			// TODO
		}
	}
	
    template<typename T>
    void display_vector(std::vector<T>& vector, char const*const msg) {
        std::cout << msg;
        for(auto i :vector)
            std::cout << i << " ";
        std::cout << std::endl;
    }

};