#pragma once
#include "matrix.hpp"
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace parallel {
inline int rank = 0, size = 1, threads = 1;
inline const char* mode() {
#ifdef USE_MPI
    return "mpi";
#elif defined(_OPENMP)
    return "openmp";
#else
    return "sequential";
#endif
}
inline void barrier() {
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}
inline double maximum(double value) {
#ifdef USE_MPI
    double result;
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
#else
    return value;
#endif
}
// broadcast=true sends initial weights; false adds the gradients from all ranks.
// MPI needs a flat array, so pack the matrix before communicating.
inline void exchange(Matrix& matrix, bool broadcast) {
#ifdef USE_MPI
    std::vector<double> buffer;
    for (const auto& row : matrix.getData()) buffer.insert(buffer.end(), row.begin(), row.end());
    if (broadcast) MPI_Bcast(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else MPI_Allreduce(MPI_IN_PLACE, buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    size_t index = 0;
    for (auto& row : matrix.getData()) for (auto& value : row) value = buffer[index++];
#endif
}
// Start MPI when this object is created and finish it when main exits.
struct Runtime {
    Runtime(int& argc, char**& argv) {
#ifdef USE_MPI
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    }
    ~Runtime() {
#ifdef USE_MPI
        MPI_Finalize();
#endif
    }
};
}

