#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include <iostream>
#include "student.h"

namespace
{

    /**
     * @class PrimeTwinsMonitor
     * @brief A thread-safe monitor for storing prime twin pairs.
     *
     * This class provides a thread-safe mechanism for storing prime twin pairs.
     * It uses a mutex to ensure that the addPrime operation is atomic and thread-safe.
     */
    class PrimeTwinsMonitor
    {
    public:
        /**
         * @brief Default constructor.
         */
        PrimeTwinsMonitor() : primes() {}

        /**
         * @brief Adds a prime twin pair to the monitor.
         * @param prime The prime twin pair to add.
         */
        void addPrime(std::pair<uint32_t, uint32_t> prime) {
            std::lock_guard<std::mutex> lock(mutex);
            primes.push_back(prime);
        }

        /**
         * @brief Returns the prime twin pairs stored in the monitor.
         * @return A vector of prime twin pairs.
         */
        std::vector<std::pair<uint32_t, uint32_t>> getPrimes() const {
            return primes;
        }

    private:
        std::mutex mutex; ///< Mutex for ensuring thread safety.
        std::vector<std::pair<uint32_t, uint32_t>> primes; ///< Vector for storing prime twin pairs.
    };

    /**
    * @class IntervalMonitor
    * @brief A thread-safe monitor for managing a range of integers.
    *
    * This class provides a thread-safe mechanism for managing a range of integers.
    * It uses a mutex to ensure that the getNumber operation is atomic and thread-safe.
    */
    class IntervalMonitor
    {
    public:
        IntervalMonitor(uint32_t min, uint32_t max)
                : val(min), max(max) {}

        int getNumber() {
            std::lock_guard<std::mutex> lock(mutex);
            int result = val;
            val++;
            return result;
        }

        uint32_t getMax() const{
            return max;
        }


    private:
        std::mutex mutex;
        uint32_t val;
        uint32_t max;
    };

    bool are_2_pairs_sorted(const std::pair<uint32_t,uint32_t>& a, const std::pair<uint32_t,uint32_t>& b) {
        return std::get<0>(a) < std::get<0>(b);
    }

    bool is_prime(const uint32_t n) {
        // check division from 2 to n (not efficient at all!)
        for (uint32_t d = 2; d < n; ++d)
            if ((n % d) == 0 ) // d is a divisor, n is not prime
                return false;
        // we have not found any divisor: n is prime
        return true;
    }

    /**
      * @brief Calculates and stores prime twin pairs in a given range.
      *
      * This function calculates prime twin pairs in a range of numbers.
      * It uses an IntervalMonitor to get the next number to check, and a PrimeTwinsMonitor to store the prime twin pairs.
      * The function runs in a loop until it has checked all numbers up to the maximum value in the interval.
      * If a number and its successor (num + 2) are both prime, they form a prime twin pair and are added to the PrimeTwinsMonitor.
      *
      * @param interval_monitor Reference to an IntervalMonitor object that manages the range of numbers to check.
      * @param prime_monitor Reference to a PrimeTwinsMonitor object that stores the prime twin pairs.
    */
    void calculate(IntervalMonitor& interval_monitor, PrimeTwinsMonitor& prime_monitor) {
        int num = interval_monitor.getNumber(); // Get the next number to check from the interval monitor.
        const uint32_t max = interval_monitor.getMax(); // Get the maximum value in the interval.
        while (num <= max) { // Continue until all numbers in the interval have been checked.
            if (is_prime(num) && is_prime(num + 2)) // If a number and its successor (num + 2) are both prime, they form a prime twin pair.
                prime_monitor.addPrime({num, num + 2}); // Add the prime twin pair to the prime monitor.
            num = interval_monitor.getNumber(); // Get the next number to check.
        }
    }


}

bool StudentWork5::isImplemented() const {
    return true;
}

std::vector<std::pair<uint32_t,uint32_t>>
StudentWork5::run(const uint32_t min, const uint32_t max, const uint32_t nb_threads)
{
    std::vector<std::pair<uint32_t, uint32_t>> result;
    PrimeTwinsMonitor prime_monitor;
    IntervalMonitor interval_monitor(min, max);

    std::cout << "starting " << nb_threads << " threads ..." << std::endl;
    std::vector<std::thread> threads(nb_threads);

    for (int i = 0; i < nb_threads; ++i)
        threads[i] = std::thread(calculate, std::ref(interval_monitor), std::ref(prime_monitor));

    // synchronize threads:
    for (int i = 0; i < nb_threads; ++i)
        threads[i].join();

    result = prime_monitor.getPrimes();

    std::cout << "threads have completed." << std::endl;

    std::sort(result.begin(), result.end(), are_2_pairs_sorted);

    return result;
}
