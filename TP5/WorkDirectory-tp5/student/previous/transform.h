#pragma once

#include <OPP.h>
#include <vector>
#include <thread>
#include <algorithm>


namespace OPP
{
    template<   typename InputIteratorType,
            typename OutputIteratorType,
            typename MapFunction>
    inline
    void transform(
            const InputIteratorType&& aBegin, // left operand
            const InputIteratorType&& aEnd,
            OutputIteratorType&& oBegin, // destination
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
            for (auto iter = aBegin; iter < aEnd; ++iter)
                oBegin[iter - aBegin] = functor(*iter);
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
                    [start, end, aBegin, oBegin, functor](){
                        for (auto iter = start; iter < end; ++iter)
                            oBegin[iter] = functor(aBegin[iter]);
                    }
            ));
        }

        // Wait for all tasks to complete
        for (auto &&future : futures)
            future.get();
    }


    // second version: two input iterators!
    template<   typename InputIteratorType,
            typename OutputIteratorType,
            typename MapFunction>
    inline
    void transform(
            const InputIteratorType&& aBegin, // left operand
            const InputIteratorType&& aEnd,
            const InputIteratorType&& bBegin, // right operand
            OutputIteratorType&& oBegin, // destination
            const MapFunction&& functor // binary functor
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
                oBegin[iter - aBegin] = functor(*iter, bBegin[iter - aBegin]);
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
                    [start, end, aBegin, bBegin, oBegin, functor](){
                        for (auto iter = start; iter < end; ++iter)
                            oBegin[iter] = functor(aBegin[iter], bBegin[iter]);
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