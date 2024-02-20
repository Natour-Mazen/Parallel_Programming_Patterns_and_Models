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

        Barrier barrier(realNbThreads);
        
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
            // TODO : travailler entre begin et end (des entiers) dans les tableaux
            // NB : ici tout ce qui est visible dans la fonction partition est capturé (donc utilisable)
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