#pragma once
#include <thread>
#include <vector>
#include <OPP.h>
#include <algorithm>

// scatter is a permutation of data. The destination index is given thanks to an iterator. 

namespace OPP {
        
    template<   typename InputIteratorType, 
                typename MapIteratorType, 
                typename OutputIteratorType>
        inline
    void scatter(
        const InputIteratorType&& aBegin, // left operand
        const InputIteratorType&& aEnd,
        const MapIteratorType&& map, // source index
        OutputIteratorType&& oBegin // destination
    ) {
        // TODO: a map using OPP::nbThreads threads that does something like:
        //for(auto iter = aBegin; iter<aEnd; ++iter)
        //   oBegin[map[iter-aBegin]] = iter[iter-aBegin];
        
        // chunk size
        auto fullSize = aEnd - aBegin;
        auto chunkSize = (fullSize + nbThreads-1) / nbThreads;
        // launch the threads
        std::vector<std::thread> threads(nbThreads);

        // ...
    }
};