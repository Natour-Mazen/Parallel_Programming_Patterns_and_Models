#pragma once

#include <OPP.h>
#include <StudentWork.h>
#include <previous/exclusive_scan.h>
#include <previous/inclusive_scan.h>
#include <previous/scatter.h>
#include <previous/transform.h>

#include <iostream>
#include <vector>

class StudentWorkImpl : public StudentWork {
 public:
  bool isImplemented() const;

  StudentWorkImpl() = default;
  StudentWorkImpl(const StudentWorkImpl&) = default;
  ~StudentWorkImpl() = default;
  StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

  /**
  * @brief Extracts and inverts the specified bit from a value.
  *
  * @param value The value from which to extract the bit.
  * @param bitPosition The position of the bit to extract and invert.
  * @return T The extracted and inverted bit.
  */
  template <typename T>
  inline T extractAndInvertBit(const T& value, const T& bitPosition){
      return 1 - ((value >> bitPosition) & 0x1u);
  }

    template<typename T>
    void run_radixSort_parallel(const std::vector<T>& input,
                                std::vector<T>& output) {
        const unsigned size = input.size();

        // Define a wrapper for vector references to use in swapping
        using wrapper = std::reference_wrapper<std::vector<T>>;
        // Create two vectors: output and temp, and create wrappers for them
        std::vector<T> temp(size);
        wrapper array[2] = {wrapper(output), wrapper(temp)};
        // Copy the input vector to the output vector
        std::copy(input.begin(), input.end(), output.begin());

        // Define vectors to store indices for downsweep and upsweep
        std::vector<unsigned> indicesDown(size);
        std::vector<unsigned> indicesUp(size);

        // Iterate through each bit of the elements
        for (T bitNumber = 0; bitNumber < sizeof(T) * 8; ++bitNumber) {
            const int ping = bitNumber & 1;
            const int pong = 1 - ping;

            // Get the current array being processed
            const std::vector<T>& currentArray = array[ping].get();

            // Define a transform iterator for bit extraction and inversion
            const OPP::TransformIterator predicate = OPP::make_transform_iterator(
                    currentArray.begin(), std::function([this, &bitNumber](const T& value) -> T {
                        return extractAndInvertBit(value, bitNumber);
                    }));

            // Perform exclusive scan (prefix sum) on the predicate
            OPP::exclusive_scan(predicate + 0, predicate + size, indicesDown.begin(),
                                std::plus<T>(), T(0));

            // Define a transform iterator for reversing and inverting the predicate
            const OPP::TransformIterator not_predicate_reversed = OPP::make_transform_iterator(
                    OPP::CountingIterator(1l),
                    std::function([&predicate, &size](const T& a) -> T {
                        return 1 - predicate[size - a];
                    }));

            // Perform inclusive scan (prefix sum) on the reversed predicate
            OPP::inclusive_scan(not_predicate_reversed + 0,
                                not_predicate_reversed + size, indicesUp.rbegin(),
                                std::plus<T>());

            // Scatter the elements according to the computed indices
            OPP::scatter(
                    currentArray.begin(), currentArray.end(),
                    OPP::make_transform_iterator(
                            OPP::CountingIterator(0l),
                            std::function([&predicate, &indicesDown, &indicesUp, &size](const T& a) -> T {
                                if (predicate[a])
                                    return indicesDown[a];
                                return size - indicesUp[a];
                            })),
                    array[pong].get().begin());
        }
    }

  void check();
};
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/