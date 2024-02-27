#pragma once
#include <OPP.h>

#include <algorithm>
#include <iterator>
#include <thread>
#include <vector>

// inclusive scan

namespace OPP {

    /**
     * @brief Performs an inclusive scan operation on an input range, using a specified binary function.
     *
     * @param aBegin The beginning of the input range.
     * @param aEnd The end of the input range (excluded).
     * @param oBegin The beginning of the output range.
     * @param functor The binary function to be used in the scan operation. It should be associative.
     *
     * This function performs an inclusive scan operation on the input range [aBegin, aEnd),
     * using the binary function `functor`. The result is written to the output range beginning at `oBegin`.
     * The scan operation is inclusive, meaning that the i-th output element is the result of applying
     * the binary function to the first i+1 input elements.
     */
    template <typename InputIteratorType,
              typename OutputIteratorType,
              typename BinaryFunction>
    inline void inclusive_scan_seq(
        const InputIteratorType &&aBegin,   // input begin
        const InputIteratorType &&aEnd,     // input end (excluded)
        const OutputIteratorType &&oBegin,  // output begin
        const BinaryFunction &&functor      // should be associative
    ) {
      const int fullSize = aEnd - aBegin;
      oBegin[0] = aBegin[0];

      for (int i = 1; i < fullSize; ++i)
        oBegin[i] = functor(oBegin[i - 1], aBegin[i]);
    }

    /**
     * @brief Performs an inclusive scan operation on an input range, using a specified binary function.
     * This version of the function is parallelized and uses a thread pool for improved performance on large inputs.
     *
     * @param aBegin The beginning of the input range.
     * @param aEnd The end of the input range (excluded).
     * @param oBegin The beginning of the output range.
     * @param functor The binary function to be used in the scan operation. It should be associative.
     *
     * This function performs an inclusive scan operation on the input range [aBegin, aEnd),
     * using the binary function `functor`. The result is written to the output range beginning at `oBegin`.
     * The scan operation is inclusive, meaning that the i-th output element is the result of applying
     * the binary function to the first i+1 input elements.
     *
     * If the size of the input range is less than the number of threads available,
     * the function falls back to a sequential scan operation.
     *
     * Otherwise, the function divides the input range into chunks and processes each chunk in parallel.
     * The results of the scan operation on each chunk are then combined to produce the final output.
     */
    template <typename InputIteratorType,
              typename OutputIteratorType,
              typename BinaryFunction>
    inline void inclusive_scan(
        const InputIteratorType &&aBegin,   // input begin
        const InputIteratorType &&aEnd,     // input end (excluded)
        const OutputIteratorType &&oBegin,  // output begin
        const BinaryFunction &&functor      // should be associative
    ) {
      const unsigned int nbTasks = OPP::nbThreads * 4;

      const int fullSize = aEnd - aBegin;

      const int chunkSize = (fullSize + nbTasks - 1) / nbTasks;

      OPP::ThreadPool &pool = OPP::getDefaultThreadPool();

      std::vector<std::shared_future<void>> futures;

      if (fullSize < nbTasks) {
        inclusive_scan_seq(std::move(aBegin), std::move(aEnd), std::move(oBegin),
                           std::move(functor));
        return;
      }

      for (int i = 0; i < nbTasks; ++i) {
        const int start = i * chunkSize;
        const int last = std::min(start + chunkSize, fullSize);

        if (start >= last)
            break;

        futures.emplace_back(
            std::move(pool.push_task([start, last, aBegin, oBegin, functor]() {
              inclusive_scan_seq(aBegin + start, aBegin + last, oBegin + start,
                                 std::move(functor));
            })));
      }

      for (auto &&future : futures)
          future.get();

      std::vector<typename InputIteratorType::value_type> aux(nbTasks - 1);

      aux[0] = oBegin[chunkSize - 1];

      for (int i = 1; i < nbTasks - 1; ++i)
        aux[i] = functor(aux[i - 1], oBegin[chunkSize * (i + 1) - 1]);

      futures.clear();

      for (int i = 0; i < nbTasks - 1; ++i) {
        const int start = (i + 1) * chunkSize;

        futures.emplace_back(std::move(
            pool.push_task([i, start, chunkSize, fullSize, aux, oBegin, functor]() {
              for (int j = 0; j < chunkSize; ++j) {
                if (start + j < fullSize)
                  oBegin[start + j] = functor(aux[i], oBegin[start + j]);
              }
            })));
      }

      for (auto &&future : futures)
          future.get();
    }
};
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/