#pragma once
#include <OPP.h>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <ranges>

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
        // Number of tasks is four times the number of threads
        unsigned int nbTasks = OPP::nbThreads * 4;

        // Calculate the size of the full range
        auto fullSize = aEnd - aBegin;
        // Calculate the size of each chunk, which is the full size divided by the number of threads
        auto chunkSize = (fullSize + OPP::nbThreads-1) / OPP::nbThreads;

        // If the full size is less than the number of tasks, apply the functor to each element directly
        if (fullSize < nbTasks) {
            for (auto iter = aBegin; iter < aEnd; ++iter)
                oBegin[map[iter - aBegin]] = aBegin[iter - aBegin];
            return;
        }
        // Create a vector to hold the futures returned by the tasks
        std::vector<std::shared_future<void>> futures;

        // Launch tasks
        for (int i = 0; i < nbTasks; ++i) {
            auto start = i * chunkSize;
            auto end = std::min(start + chunkSize, fullSize);

            // If the start index is greater than or equal to the end index, break the loop
            if (start >= end)
                break;

            // Add a task to the thread pool. The task applies the functor to a range of elements
            futures.emplace_back(OPP::getDefaultThreadPool().push_task(
                    [start, end, aBegin, oBegin, map](){
                        for (auto iter = start; iter < end; ++iter)
                            oBegin[map[iter]] = aBegin[iter];
                    }
            ));
        }

        // Wait for all tasks to complete
        for (auto &&future : futures)
            future.get();

    }
};
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/