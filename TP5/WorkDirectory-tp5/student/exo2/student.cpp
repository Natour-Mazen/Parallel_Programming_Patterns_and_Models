#include <iostream>
#include <functional>
#include <exo2/student.h>
#include <algorithm>

namespace 
{
    /**
     * @brief Extracts and inverts the specified bit from a value.
     *
     * @param value The value from which to extract the bit.
     * @param bitPosition The position of the bit to extract and invert.
     * @return unsigned The extracted and inverted bit.
     */
    unsigned extractAndInvertBit(unsigned value, unsigned bitPosition){
        return 1 - ((value >> bitPosition) & 0x1u);
    }

    /**
     * @brief Partitions the input vector based on the predicate vector sequentially.
     *
     * @param input The input vector to be partitioned.
     * @param predicate The predicate vector used for partitioning.
     * @param output The resulting partitioned vector.
     */
    void run_partition_sequential(
            std::vector<unsigned>& input,
            std::vector<unsigned>& predicate,
            std::vector<unsigned>& output
    ) {
        int n = 0;
        for (int i = 1; i >= 0; --i) {
            for (size_t j = 0; j < input.size(); ++j) {
                if (predicate[j] == i)
                    output[n++] = input[j];
            }
        }
    }
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_radixSort_sequential(
	std::vector<unsigned>& input,
	std::vector<unsigned>& output
) {
    // Define a wrapper for vector references
    using wrapper = std::reference_wrapper<std::vector<unsigned>>;

    // Create a temporary vector with the same size as the input
    std::vector<unsigned> temp(input.size());

    // Create wrapper references to 'output' and 'temp' vectors
    wrapper T[2] = { wrapper(output), wrapper(temp) };

    // Create a vector to store predicates
    std::vector<unsigned> predicate(input.size());

    // Copy the input vector to the output vector
    std::copy(input.begin(), input.end(), output.begin());

    // Iterate through each bit of the unsigned integer (assuming 32 bits for simplicity)
    for(unsigned bitNumber = 0; bitNumber < sizeof (unsigned) * 8 ; ++bitNumber)
    {
        // Determine the indexes of the ping and pong arrays for swapping
        const int ping = bitNumber & 1;
        const int pong = 1 - ping;

        // Calculate the predicate for each element based on the current bit
        for (size_t i = 0; i < input.size(); ++i)
            predicate[i] = extractAndInvertBit(T[ping].get()[i], bitNumber);

        // Partition the input based on the predicate, storing the result in the pong array
        run_partition_sequential(T[ping], predicate, T[pong]);
    }
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/