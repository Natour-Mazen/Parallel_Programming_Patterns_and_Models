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
        // TODO: a map using OPP::nbThreads threads
        //  that does something like the following
        //for(auto iter = aBegin; iter<aEnd; ++iter)
        //    oBegin[iter-aBegin] = functor(*iter);
        
        // chunk size
        auto fullSize = aEnd - aBegin;
        auto chunkSize = (fullSize + OPP::nbThreads-1) / OPP::nbThreads;
        // launch the threads
        std::vector<std::thread> threads(OPP::nbThreads);
        // ...
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
        // TODO: a map using OPP::nbThreads threads
        // that does something like:
        //for(auto iter = aBegin; iter<aEnd; ++iter)
        //   oBegin[iter-aBegin] = functor(*iter, bBegin[iter-aBegin]);
        
        // chunk size
        auto fullSize = aEnd - aBegin;
        auto chunkSize = (fullSize + OPP::nbThreads-1) / OPP::nbThreads;
        // launch the threads
        std::vector<std::thread> threads(OPP::nbThreads);
        // ...
    }
};