#include <mpi.h>
#include <1-Resolution/Resolution.h>
#include <algorithm>

void Solve(
    const OPP::MPI::Communicator& communicator,
    const DistributedRowMatrix& L,
    const DistributedBlockVector& B,
    DistributedBlockVector& X,
    const int N
) {
    OPP::MPI::Ring ring(communicator.communicator);
    
    // Here, we have a block of row (take care to the distribution!)
    // block size ... or B.End() - B.Start() except the last processor (it can be smaller for last block)
    const int m = (N+ring.getSize()-1) / ring.getSize(); 
    // check it is ok
    if( m < B.End() - B.Start() )
        std::cerr << "Bad value for m="<<m << std::endl;
        
    // TODO
}