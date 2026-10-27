# MNIST parallel programming assignment

Recognize handwritten digits using one model in three modes: sequential,
OpenMP (CPU threads), and MPI (processes across laptops).

Source: [nisbenz/ScratchNet](https://github.com/nisbenz/ScratchNet).
The original LICENSE remains in ScratchNet. Our changes add parallel training
and simplify configuration, loading, and output.

## Build and run

Open a VS Code PowerShell terminal in **ScratchNet**. C++ Build Tools and the
Microsoft MPI SDK/runtime must be installed. The script sets up the compiler.

~~~powershell
.\build.ps1

# Sequential: five epochs.
.\build\neural_network.exe 5

# OpenMP: five epochs and four threads.
.\build\neural_network_omp.exe 5 4

# MPI: four processes on THIS laptop.
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -n 4 .\build\neural_network_mpi.exe 5
~~~

| Argument | Default | Meaning |
|---|---|---|
| First | 5 | Epochs: passes through the training set |
| Second | 1 | Threads; more than 1 only for OpenMP |
| Third | ../../MNIST | Dataset folder |

Fixed settings in main.cpp: batch 64, learning rate 0.01, seed 42,
network 784 -> 128 -> 64 -> 10. Keep them identical for comparisons.
Each command performs one run; no automatic sweeps or CSV exports.

## Input and output

Input: four binary MNIST IDX files, containing 60,000 training images and
10,000 test images. Each image has 28 x 28 pixels, normalized to 0–1.

Output: one short configuration block, training time, correct predictions,
and test accuracy. Take a screenshot for the report, or save the output:

~~~powershell
.\build\neural_network.exe 5 | Tee-Object sequential.txt
~~~

Timing covers training only, excluding loading, initialization, and testing.
MPI includes gradient communication and reports the slowest process's time.
Collect fresh timings: old CSVs used a different timing definition.

## How it works

- Sequential: calculate the batch gradients and update the weights.
- OpenMP: split matrix multiplication output rows among CPU threads.
- MPI: split each batch among processes, add gradients with MPI_Allreduce,
  then apply the same update on every process. Batch 64 means 64 images total.

Initial weights and image order are shared for consistent MPI training.
Key files to discuss: src/main.cpp (load/train/time/test),
src/neural_network.cpp (training and gradients), src/matrix.cpp (OpenMP),
and include/parallel.hpp (MPI helpers). Comments explain these steps.
Other C++ files provide the model's layers, activations, and data loader.

## Four laptops

Each laptop needs the same MS-MPI runtime, dataset, and executable, and working
network/account/firewall access with the MS-MPI Launch Service.
Follow [Microsoft's launch guide](https://github.com/microsoft/Microsoft-MPI/blob/master/docs/RunningMSMPI.md).

On EVERY laptop place neural_network_mpi.exe in C:\MpiMNIST and the four data
files in C:\MpiMNIST\MNIST. From one laptop, replace these hostnames and run:

~~~powershell
& 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe' -hosts 4 LAPTOP1 1 LAPTOP2 1 LAPTOP3 1 LAPTOP4 1 -wdir C:\MpiMNIST C:\MpiMNIST\neural_network_mpi.exe 5 1 C:\MpiMNIST\MNIST
~~~

Try two laptops before four, with one process per laptop. Rank 0 prints results.
Local MPI tests do not verify the four-laptop network.

## Results and report

Run each configuration three times; use the median (middle time).
Try 1, 2, 4 OpenMP threads and 1, 2, 4 MPI laptops. Keep all training settings
fixed and use the same laptop for the sequential baseline. Record machine
specifications and network details.

| Mode | Workers | Median seconds | Accuracy % | Speedup | Efficiency % |
|---|---:|---:|---:|---:|---:|
| Sequential | 1 | Measured | Measured | 1.00 | — |
| OpenMP | 2 | Measured | Measured | Calculate | Calculate |
| OpenMP | 4 | Measured | Measured | Calculate | Calculate |
| MPI | 2 | Measured | Measured | Calculate | Calculate |
| MPI | 4 | Measured | Measured | Calculate | Calculate |

Speedup = sequential time / parallel time.
Efficiency = speedup / workers x 100%.
Workers means threads for OpenMP and laptops for one-process-per-laptop MPI.
Parallel runs may be slower; explain the overhead using your measured results.

Keep the report within 10 pages: problem/model, libraries for multicore/network/
GPU/cloud platforms, input/output, key code, results, speedup/efficiency, and
references. Include the requested code appendix; clarify whether it counts
toward 10 pages. Submit the programs and a three-minute demonstration too.

## Folders

- ../MNIST: input data (beside Neural-Network-MNIST-Sample).
- ScratchNet/src and include: commented source.
- ScratchNet/build.ps1: build all three programs.
- ScratchNet/build: generated programs; keep it to run without rebuilding.
- ScratchNet/LICENSE: original license.
- Testing: OpenMP and MPI setup smoke tests; see [testing instructions](Testing/README.md).

Run commands from ScratchNet so the default dataset path resolves correctly.
Use one epoch instead of five for a quick end-to-end check, then use the same
epoch count for every configuration in the report. Setup smoke tests do not
measure MNIST accuracy or establish performance speedup.
