#include <thread> // C++ 11
#include <mutex> // C++ 11
#include <iostream>
#include <cmath>
#include <vector>
#include <exo4/student.h>

namespace {

    class Monitor
    {
    public:
        Monitor(): result(0.0){}

        void addToResult(double value) {
            std::lock_guard<std::mutex> lock(mutex);
            result += value;
        }

        double getResult() const { return result; }

    private:
        std::mutex mutex;
        double result;
    };

    const unsigned int limit = 1 << 28; // 2^28 == 256 millions

    void calculate(int num, const unsigned nb_threads, Monitor& monitor) {
        double acc = 0.0;
        for (unsigned int n = num; n < limit; n += nb_threads)
            acc += pow(-1.0, n) / (2.0 * n + 1.0);
        monitor.addToResult(acc);
    }
}

bool StudentWork4::isImplemented() const {
    return true;
}

/// nb_threads is between 1 to 64 ...
double StudentWork4::run(const unsigned nb_threads)
{
    std::cout << "starting " << nb_threads << " threads ..." << std::endl;
    std::vector<std::thread> threads(nb_threads);
    Monitor monitor;

    for (int i = 0; i < nb_threads; ++i)
        threads[i] = std::thread(calculate, i, nb_threads,std::ref(monitor));

    // synchronize threads:
    for (int i = 0; i < nb_threads; ++i)
        threads[i].join();

    std::cout << "threads have completed." << std::endl;
    double pi = monitor.getResult() * 4.0;

    std::cout.precision(12);
    std::cout << "our evaluation is: " << std::fixed << pi << std::endl;

    return pi;
}
