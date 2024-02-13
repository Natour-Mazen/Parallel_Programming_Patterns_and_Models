#include <immintrin.h> //AVX+SSE Extensions#include <vector>
#include <cmath>
#include <iostream>
#include <exo4/student.h>
#include <thread>
#include <vector>

namespace {
	struct Convertor 
	{ 
		union {
			__m256 avx; 
			float f[8];
		} u;
		// constructor
		Convertor(const __m256& m256) { u.avx = m256; };
		// accessor to element i (between 0 and 7 included)
		float operator()(int i) const 
		{ 			
			return u.f[i]; 
		}
		float& operator()(int i) 
		{ 			
			return u.f[i]; 
		}
		// prints data on a given stream
		friend std::ostream& operator<<(std::ostream&, const Convertor&c);
	};

	std::ostream& operator<<(std::ostream& os, const Convertor&c) 
	{
		os << "{ ";
		for(int i=0; i<7; ++i) {
			os << c(i) << ", ";
		}
		return os << c(7) << " }";
	}

    // This function calculates the sum of a portion of an array.
    void partial_sum(const float* input, size_t start, size_t end, float* result) {
        // Initialize the result to 0.0
        *result = 0.0;

        // Iterate over the portion of the array from start to end
        for (size_t i = start; i < end; ++i) {
            // Add each element to the result
            *result += input[i];
        }
    }
}

bool StudentWorkImpl::isImplemented() const {
	return !false;
}

// calculate with floats
#pragma optimize("", off)
float StudentWorkImpl::run(float const * const input, const size_t size)
{
    // If the size is zero, return 0.0
    if (size == 0)
        return 0.0;

    // the number of threads
    size_t num_threads = 8;

    // Create a vector to hold the threads
    std::vector<std::thread> threads(num_threads);

    // Create a vector to hold the results from each thread
    std::vector<float> results(num_threads);

    // Calculate the size of the chunk of data each thread will process
    size_t chunk_size = size / num_threads;

    // Launch the threads
    for (size_t i = 0; i < num_threads; ++i) {
        // Calculate the start and end indices for this thread's chunk of data
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? size : start + chunk_size;

        // Launch the thread, passing it the function to execute and the arguments
        threads[i] = std::thread(partial_sum, input, start, end, &results[i]);
    }

    // Wait for all the threads to finish
    for (auto& t : threads) {
        t.join();
    }

    // Sum up the results from each thread
    float sum = 0.0;
    for (const auto& r : results) {
        sum += r;
    }

    // Return the total sum
    return sum;
}

// calculate with mm256
#pragma optimize("", off)
float StudentWorkImpl::run(__m256 const *const input, const size_t size)
{
    // If the size is zero, return 0.0
    if (size == 0)
        return 0.0;

    // Start from the last element
    size_t i = size - 1;
    // Initialize the sum with the last element
    __m256 sum = input[i];

    // Add up all the elements from the end to the start
    while (i-- > 0)
        sum = _mm256_add_ps(sum, input[i]);

    /*
     * Here we perform a dot product between our AVX which contains the sums and an AVX filled with 1, then we put a
     * mask equal to 241 which is 1111 0001 in binary which will finally store a part of the sum in case 0
     * of our AVX and the other part of the sum in case 4.
    */
    __m256 dp = _mm256_dp_ps(sum, _mm256_set1_ps(1.0), 241);

    // Convert the result to float
    Convertor c = Convertor(dp);

    // Return the sum
    return c(0) + c (4);
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/