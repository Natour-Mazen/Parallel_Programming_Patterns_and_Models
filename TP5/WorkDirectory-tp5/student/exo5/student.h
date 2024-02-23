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

    template <typename T>
    inline unsigned extractAndInvertBit(const T& value, const T& bitPosition){
        return 1 - ((value >> bitPosition) & 0x1u);
    }

	template<typename T>
	void run_radixSort_parallel(
		std::vector<T>& input,
		std::vector<T>& output
	) {
		std::copy(input.begin(), input.end(), output.begin());
		std::vector<T> temp(input.size());
		std::vector<T>* array[2] = { &output, &temp }; // des pointeurs conviennent aussi !

        std::vector<unsigned> predicate(input.size());
        std::copy(input.begin(), input.end(), output.begin());

		for(unsigned numeroBit=0; numeroBit<sizeof(T)*8; ++numeroBit) 
		{
            const int ping = numeroBit & 1;
            const int pong = 1 - ping;

            OPP::transform(array[ping]->begin(), array[ping]->end(),
                           predicate.begin(),
                           std::function([this, &numeroBit](const T& value) -> T {
                               return extractAndInvertBit(value, numeroBit);
                           }));

            OPP::partition(array[ping]->begin(), array[ping]->end(),
                           predicate.begin(), array[pong]->begin());
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