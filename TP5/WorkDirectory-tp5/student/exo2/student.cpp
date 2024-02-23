#include <iostream>
#include <functional>
#include <exo2/student.h>
#include <algorithm>

namespace 
{
    unsigned extractAndInvertBit(unsigned value, unsigned bitPosition){
        return 1 - ((value >> bitPosition) & 0x1u);
    }

    void run_partition_sequential(
            std::vector<unsigned>& input,
            std::vector<unsigned>& predicate,
            std::vector<unsigned>& output
    ) {
        int n = 0;
        for (int i = 1; i >= 0; --i) {
            for (size_t j = 0; j < input.size(); ++j) {
                if (predicate[j] == i)
                    output[n++] = input[j];
            }
        }
    }
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_radixSort_sequential(
	std::vector<unsigned>& input,
	std::vector<unsigned>& output
) {
	// utiliser l'algorithme vu en court/TD
	// pour chaque bit, en partant du poids faible
	//   calculer predicat = ième bit (c'est un MAP, séquentiel ici)
	//   partitionner (séquentiellement)
	// ... et c'est tout !
	// Attention quand même : le partitionnement nécessite un tableau auxiliaire !!!
	// Le plus simple est d'utiliser un nouveau tableau plus output (qui reçoit une copie de input)
	using wrapper = std::reference_wrapper<std::vector<unsigned>>;

	std::vector<unsigned> temp(input.size());

	wrapper T[2] = { wrapper(output), wrapper(temp) };

	std::vector<unsigned> predicate(input.size());

	std::copy(input.begin(), input.end(), output.begin());

	for(unsigned numeroBit=0; numeroBit<sizeof(unsigned)*8; ++numeroBit) 
	{
		const int ping = numeroBit & 1;
		const int pong = 1 - ping;

        for (size_t i = 0; i < input.size(); ++i)
            predicate[i] = extractAndInvertBit(T[ping].get()[i], numeroBit);

        run_partition_sequential(T[ping],predicate,T[pong]);
	}
}
