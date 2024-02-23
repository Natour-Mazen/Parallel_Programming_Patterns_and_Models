#pragma once
#include <thread>
#include <vector>
#include <OPP.h>
#include <algorithm>

// partition

namespace OPP 
{

    template<   typename InputIteratorType, 
                typename PredicateIteratorType, 
                typename OutputIteratorType >
    inline
    void partition(
        const InputIteratorType&& aBegin, // input begin
        const InputIteratorType&& aEnd,   // input end (excluded)
        const PredicateIteratorType&& predBegin,   // predicate begin, should be iterator on int ...
        const OutputIteratorType&& oBegin // output begin
    ) {
        // chunk size
        const auto fullSize = static_cast<decltype(nbThreads)>(aEnd - aBegin);
        const auto realNbThreads = std::min(fullSize, nbThreads);
        const auto chunkSize = (fullSize + realNbThreads-1) / realNbThreads;

        using T = typename InputIteratorType::value_type;

        Barrier barrier1(realNbThreads);
        Barrier barrier2(realNbThreads);
        Barrier barrier3(realNbThreads);

        std::vector<typename OutputIteratorType::value_type> sumsTrue(fullSize);
        std::vector<typename OutputIteratorType::value_type> sumsFalse(fullSize);

        std::vector<typename OutputIteratorType::value_type> partialSumsTrue(realNbThreads);
        std::vector<typename OutputIteratorType::value_type> partialSumsFalse(realNbThreads);


        auto fun_thread = [&] (
            const size_t begin,
            const size_t end,
            const unsigned thread_num
        ) -> void
        {
            sumsTrue[begin] = T(0);
            sumsFalse[fullSize - 1 - begin] = 1 - predBegin[fullSize - 1 - begin];
            for (size_t i = begin + 1; i < end; ++i) {
                sumsTrue[i] = sumsTrue[i - 1] + predBegin[i - 1];
                sumsFalse[fullSize - 1 - i] = sumsFalse[fullSize - i] + 1 - predBegin[fullSize - 1 - i];
            }

            barrier1.arrive_and_wait();

            if (thread_num == 0u) {
                partialSumsTrue[0] = sumsTrue[chunkSize - 1] + predBegin[chunkSize - 1];
                partialSumsFalse[realNbThreads - 1] = sumsFalse[fullSize - chunkSize];
                for (size_t i = 1; i < realNbThreads - 1; ++i) {
                    partialSumsTrue[i] = partialSumsTrue[i - 1] + sumsTrue[chunkSize * (i + 1) - 1] + predBegin[chunkSize * (i + 1) - 1];
                    partialSumsFalse[realNbThreads - 1 - i] = partialSumsFalse[realNbThreads - i] + sumsFalse[fullSize - chunkSize * (i + 1)];
                }
                barrier2.arrive_and_wait();
            } else {
                barrier2.arrive_and_wait();
                for (size_t i = begin; i < end; ++i) {
                    sumsTrue[i] += partialSumsTrue[thread_num - 1];
                    sumsFalse[fullSize - 1 - i] += partialSumsFalse[realNbThreads - thread_num];
                }
            }

            barrier3.arrive_and_wait();

            for (size_t i = begin; i < end; ++i) {
                if (predBegin[i]) {
                    oBegin[sumsTrue[i]] = aBegin[i];
                } else {
                    oBegin[fullSize - sumsFalse[i]] = aBegin[i];
                }
            }
        };
        
        // launch the threads
        std::vector<std::thread> threads(realNbThreads);
        for(auto i=0u; i<realNbThreads; i++) {
            threads[i] = 
                std::thread(
                    fun_thread,
                    i*chunkSize, 
                    std::min((i+1)*chunkSize, fullSize), 
                    i
                );
        };
        for(auto& th : threads)
            th.join();
    }
};