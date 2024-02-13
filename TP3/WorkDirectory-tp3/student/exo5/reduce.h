#pragma once
#include <OPP.h>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <ranges>

// gather is a permutation of data. The source index is given thanks to an iterator. 

namespace OPP {
    
    template<   typename InputIteratorType, 
                typename T, 
                typename MapFunction>
        inline
    T reduce(
        const InputIteratorType&& aBegin, 
        const InputIteratorType&& aEnd,
        const T&& init,
        const MapFunction&& functor // unary functor
    ) {
        // Number of tasks is four times the number of threads
        unsigned int nbTasks = OPP::nbThreads * 4;

        // Calculate the size of the full range
        auto fullSize = aEnd - aBegin;
        // Calculate the size of each chunk, which is the full size divided by the number of threads
        auto chunkSize = (fullSize + OPP::nbThreads-1) / OPP::nbThreads;

        // If the full size is less than the number of tasks, apply the functor to each element directly
        if (fullSize < nbTasks) {
            T sum = init;
            for (auto iter = aBegin; iter < aEnd; ++iter)
                sum = functor(sum,aBegin[iter - aBegin]);
            return sum;
        }
        // Create a vector to hold the futures returned by the tasks
        std::vector<std::shared_future<T>> futures;

        // Launch tasks
        for (int i = 0; i < nbTasks; ++i) {
            auto start = i * chunkSize;
            auto end = std::min(start + chunkSize, fullSize);

            // If the start index is greater than or equal to the end index, break the loop
            if (start >= end)
                break;

            // Add a task to the thread pool. The task applies the functor to a range of elements
            futures.emplace_back(OPP::getDefaultThreadPool().push_task(
                    [start, end, aBegin, functor, init]() -> T{
                        T acc = init;
                        for (auto iter = start; iter < end; ++iter)
                            acc = functor(acc, aBegin[iter]);
                        return acc;
                    }
            ));
        }

        T sum = init;

        // Wait for all tasks to complete
        for (auto &&future : futures)
            sum += future.get();

        return sum;
    }
};
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/