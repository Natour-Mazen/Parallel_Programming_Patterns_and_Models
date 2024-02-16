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
        const unsigned int nbTasks = OPP::nbThreads * 4;

        const int fullSize = aEnd - aBegin;

        const int chunkSize = (fullSize + nbTasks - 1) / nbTasks;

        OPP::ThreadPool &pool = OPP::getDefaultThreadPool();

        std::vector<std::shared_future<void>> futures;

        for (int i = 0; i < nbTasks; ++i) {
          const int start = i * chunkSize;
          const int last = std::min(start + chunkSize, fullSize);

          if (start >= last)
            break;

          futures.emplace_back(
          pool.push_task([start, last, aBegin, oBegin, functor]() {
                //    We could have used the same function that is in the file exo1/student.h for the
                // sequential version, but I prefer to use the one defined by C++ in order to gain
                // performance.
                  std::inclusive_scan(aBegin + start, aBegin + last, oBegin + start, functor);
              }));
        }

        for (auto &&future : futures)
          future.get();

        std::vector<typename OutputIteratorType::value_type> aux(nbTasks - 1);

        aux[0] = oBegin[chunkSize - 1];

        for (int i = 1; i < nbTasks - 1; ++i)
          aux[i] = functor(aux[i - 1], oBegin[chunkSize * (i + 1) - 1]);

        futures.clear();

        for (int i = 0; i < nbTasks - 1; ++i) {
          const int start = (i + 1) * chunkSize;

          futures.emplace_back(
              pool.push_task([i, start, chunkSize, fullSize, aux, oBegin, functor]() {
                for (int j = 0; j < chunkSize; ++j) {
                  if (start + j < fullSize)
                    oBegin[start + j] = functor(aux[i], oBegin[start + j]);
                }
              }));
        }

        for (auto &&future : futures)
          future.get();
    }
};
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/