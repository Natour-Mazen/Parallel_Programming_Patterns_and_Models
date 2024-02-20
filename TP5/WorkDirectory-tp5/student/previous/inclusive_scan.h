#pragma once
#include <thread>
#include <vector>
#include <OPP.h>
#include <algorithm>
#include <iterator>

// inclusive scan

namespace OPP {
    
    template<   typename InputIteratorType, 
                typename OutputIteratorType, 
                typename BinaryFunction >
    inline
    void inclusive_scan(
        const InputIteratorType&& aBegin, // input begin
        const InputIteratorType&& aEnd,   // input end (excluded)
        const OutputIteratorType&& oBegin, // output begin
        const BinaryFunction&& functor // should be associative
    ) {
        // size of the input
        const auto fullSize = static_cast<decltype(nbThreads)>(aEnd - aBegin);
        // number of threads, even for very small input size
        const auto realNbThreads = std::min(fullSize, nbThreads);
        // chunk size (or block size)
        const auto chunkSize = (fullSize + realNbThreads-1) / realNbThreads;
        // this allows to store one data per block
        std::vector<typename OutputIteratorType::value_type> partial_sums(realNbThreads);
        // barrier to synchronize threads ...
        Barrier barrier(realNbThreads);

        // TODO ...

    }
};