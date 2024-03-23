#include <OPP/MPI/OPP_MPI.h>

namespace
{
  int receiveOrInitIfFirstProc(const OPP::MPI::Ring &ring)
  {
    // TODO
    return -1; // change it!
  }

  void sendToNextProcIfNotLast(const OPP::MPI::Ring &ring, int number)
  {
    // TODO
  }

  int test(const OPP::MPI::Communicator &comm)
  {
    if (comm.size < 2)
    {
      std::cerr << "You must launch at least two processors ... aborting" << std::endl;
      return -1;
    }

    OPP::MPI::Ring ring(comm.communicator);

    sendToNextProcIfNotLast(ring, receiveOrInitIfFirstProc(ring));

    return 0;
  }
}

int main(int argc, char **argv)
{
  // is something missing here??

  const int err_code = test(MPI_COMM_WORLD);

  // and here?
  return err_code;
}