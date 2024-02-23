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

    template <typename T>
    inline unsigned extractAndInvertBit(const T& value, const T& bitPosition){
        return 1 - ((value >> bitPosition) & 0x1u);
    }

    template <typename T>
    void run_radixSort_parallel(const std::vector<T>& input,
                                std::vector<T>& output) {

        const unsigned int InputVectorSize = input.size();

        using VectorReference = std::reference_wrapper<std::vector<T>>;
        std::vector<T> tempVector(InputVectorSize);
        VectorReference vectors[2] = {VectorReference(output), VectorReference(tempVector)};
        std::copy(input.begin(), input.end(), output.begin());

        std::vector<unsigned int> indicesDown(InputVectorSize);
        std::vector<unsigned int> indicesUp(InputVectorSize);

        for (unsigned int bitNumber = 0; bitNumber < sizeof(T) * 8; ++bitNumber) {
            const int pingIndex = bitNumber & 1;
            const int pongIndex = 1 - pingIndex;

            const std::vector<T>& currentVector = vectors[pingIndex].get();

            const OPP::TransformIterator predicate = OPP::make_transform_iterator(
                    currentVector.begin(), std::function([&bitNumber, this](const T& value) -> T {
                        return extractAndInvertBit(value, bitNumber);
                    }));

            OPP::exclusive_scan(predicate + 0, predicate + InputVectorSize, indicesDown.begin(),
                                std::plus<T>(), T(0));

            const OPP::TransformIterator notPredicateReversed = OPP::make_transform_iterator(
                    OPP::CountingIterator(1ul),
                    std::function([&predicate, &InputVectorSize](const T& a) -> T {
                        return 1 - predicate[InputVectorSize - a];
                    }));

            OPP::inclusive_scan(notPredicateReversed + 0,
                                notPredicateReversed + InputVectorSize, indicesUp.rbegin(),
                                std::plus<T>());

            OPP::scatter(
                    currentVector.begin(), currentVector.end(),
                    OPP::make_transform_iterator(
                            OPP::CountingIterator(0ul),
                            std::function([&predicate, &indicesDown, &indicesUp, &InputVectorSize](const T& a) -> T {
                                if (predicate[a])
                                    return indicesDown[a];
                                return InputVectorSize - indicesUp[a];
                            })),
                    vectors[pongIndex].get().begin());
        }
    }

    void check();
};