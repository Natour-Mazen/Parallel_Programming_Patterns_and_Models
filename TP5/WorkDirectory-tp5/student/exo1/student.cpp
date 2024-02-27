#include <iostream>
#include <exo1/student.h>


namespace {
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}
/**
 * @brief Partition the input vector sequentially based on the predicate vector.
 *
 * This function partitions the input vector into two partitions based on the predicate vector.
 * The elements for which the corresponding predicate value is 0 are placed before the elements
 * for which the predicate value is 1.
 *
 * @param input The input vector to be partitioned.
 * @param predicate The predicate vector determining the partitioning.
 * @param output The output vector where the partitioned elements will be stored.
 */
void StudentWorkImpl::run_partition_sequential(
        std::vector<int>& input,
        std::vector<int>& predicate,
        std::vector<int>& output
) {
    int n = 0;
    // Iterate over predicate values in reverse order (1 before 0)
    for (int i = 1; i >= 0; --i) {
        // Iterate over elements in input vector
        for (size_t j = 0; j < input.size(); ++j) {
            // If the predicate value matches the current value of i, add corresponding input element to output
            if (predicate[j] == i)
                output[n++] = input[j];
        }
    }
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/