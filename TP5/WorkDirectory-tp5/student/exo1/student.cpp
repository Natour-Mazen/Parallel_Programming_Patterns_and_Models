#include <iostream>
#include <exo1/student.h>


namespace {
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_partition_sequential(
	std::vector<int>& input,
	std::vector<int>& predicate,
	std::vector<int>& output
) {
    int n = 0;
    for (int i = 1; i >= 0; --i) {
        for (size_t j = 0; j < input.size(); ++j) {
            if (predicate[j] == i)
                output[n++] = input[j];
        }
    }
}
