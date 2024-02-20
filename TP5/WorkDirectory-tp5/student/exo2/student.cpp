#include <iostream>
#include <functional>
#include <exo2/student.h>
#include <algorithm>

namespace 
{
	
}

bool StudentWorkImpl::isImplemented() const {
	return false;
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
		// TODO (extraire le bit, partitionner)
	}
}
