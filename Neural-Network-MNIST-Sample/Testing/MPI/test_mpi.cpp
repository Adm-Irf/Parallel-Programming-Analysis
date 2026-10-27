// Installation/communication smoke test; no MNIST dataset is required.
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <set>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long count = rank == 0 ? 1000000 : 0;
    MPI_Bcast(&count, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    const long long first = count * rank / size + 1;
    const long long last = count * (rank + 1) / size;
    long long local = 0, total = 0;
    for (long long value = first; value <= last; ++value) local += value;
    MPI_Reduce(&local, &total, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    char hostname[MPI_MAX_PROCESSOR_NAME] = {};
    int length = 0;
    MPI_Get_processor_name(hostname, &length);
    std::vector<char> hosts(static_cast<size_t>(size) * MPI_MAX_PROCESSOR_NAME);
    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hosts.data(),
               MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    int passed = 1;
    if (rank == 0) {
        std::set<std::string> distinct;
        for (int i = 0; i < size; ++i) {
            const std::string host(hosts.data() + i * MPI_MAX_PROCESSOR_NAME);
            distinct.insert(host);
            std::cout << "Rank " << i << ": " << host << '\n';
        }
        const long long expected = count * (count + 1) / 2;
        passed = total == expected;
        std::cout << "Processes: " << size << "\nDistinct processor names: " << distinct.size()
                  << "\nSum: " << total << "\nExpected: " << expected << '\n'
                  << (passed ? "PASS" : "FAIL") << ": broadcast and distributed sum\n";
    }
    MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return passed ? 0 : 1;
}
