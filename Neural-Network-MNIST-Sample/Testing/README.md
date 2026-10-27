# MPI and OpenMP setup tests

These C++ smoke tests check your parallel programming setup before implementing
MNIST training. They do not need the dataset. Both sum integers 1 to 1,000,000;
the expected answer is **500000500000**. Success prints `PASS` and exits with 0.
Timing is diagnostic only: this tiny workload does not establish useful speedup.

## Windows prerequisites

- Install Visual Studio Build Tools with **Desktop development with C++**.
- For MPI, install the **Microsoft MPI runtime** on each laptop and the
  **Microsoft MPI SDK** on the machine where you compile.
- Open **x64 Native Tools Command Prompt for VS** after installation.
  The commands below use that command prompt (cmd.exe), not PowerShell.
- Start in the repository root, then run `cd Neural-Network-MNIST-Sample\Testing`.

Official references:
[MS-MPI downloads](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi),
[MSVC OpenMP option](https://learn.microsoft.com/en-us/cpp/build/reference/openmp-enable-openmp-2-0-support),
[MS-MPI multi-node launch guide](https://github.com/microsoft/Microsoft-MPI/blob/master/docs/RunningMSMPI.md).

## OpenMP on one laptop

```bat
cd OpenMP
cl /nologo /EHsc /O2 /openmp /MT test_openmp.cpp /Fe:test_openmp.exe
set OMP_DYNAMIC=FALSE
set OMP_NUM_THREADS=1
test_openmp.exe
set OMP_NUM_THREADS=2
test_openmp.exe
set OMP_NUM_THREADS=4
test_openmp.exe
cd ..
```

Check `Actual threads` and `PASS`. OpenMP uses cores within this laptop;
it does not communicate with the other laptops. Run this test on each laptop
independently to verify its setup.

## MPI locally first

```bat
cd MPI
cl /nologo /EHsc /O2 /MT /I"%MSMPI_INC%" test_mpi.cpp /Fe:test_mpi.exe /link /LIBPATH:"%MSMPI_LIB64%" msmpi.lib
mpiexec -n 1 test_mpi.exe
mpiexec -n 3 test_mpi.exe
mpiexec -n 4 test_mpi.exe
cd ..
```

Three processes also exercise an uneven division of the input.
Four local processes should show four ranks but one distinct processor name.
That verifies local MPI communication, not communication between laptops.

If `cl` is missing, use the x64 developer command prompt. If `mpi.h` or
`msmpi.lib` is missing, check that the SDK is installed and that `MSMPI_INC`
and `MSMPI_LIB64` are defined in a freshly opened command prompt.
If `mpiexec` is missing, check the runtime installation and its PATH entry.

## MPI across four Windows laptops

1. Install the same MS-MPI runtime version on all four laptops.
2. Connect them to a network where they can reach each other. Record each
   computer's name using `hostname`; each laptop should have a unique name.
3. Copy the compiled `test_mpi.exe` to `C:\MpiTests` on **every** laptop.
4. Start the **MS-MPI Launch Service** on each laptop using Windows Services.
   Multi-machine launching requires suitable account credentials/access;
   follow the official launch guide linked above for your environment.
5. Allow the required MPI programs through Windows Firewall on your trusted
   network. Do not disable the firewall. Some campus/guest Wi-Fi networks
   isolate clients; use a network that permits laptop-to-laptop traffic.
6. From one laptop, replace the example names below with the real hostnames:

```bat
mpiexec -hosts 4 LAPTOP1 1 LAPTOP2 1 LAPTOP3 1 LAPTOP4 1 -wdir C:\MpiTests test_mpi.exe
```

Expect four ranks, four distinct processor names, sum `500000500000`, and `PASS`.
Inspect the rank-to-host lines to confirm that all intended laptops participated.
You only launch this command once; MPI launches the remote processes.
If launch fails before any program output, troubleshoot name resolution,
service status, authentication, executable location, and network access.

## Linux alternative

With a C++ compiler, OpenMP support, and an MPI development/runtime installation,
run these from `Testing`:

```sh
g++ -O2 -fopenmp OpenMP/test_openmp.cpp -o OpenMP/test_openmp
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./OpenMP/test_openmp
mpic++ -O2 MPI/test_mpi.cpp -o MPI/test_mpi
mpiexec -n 4 ./MPI/test_mpi
```

Multi-host Linux launch configuration depends on the MPI implementation and
typically requires remote login setup and consistent executable paths.
Use a consistent operating system/MPI environment across the cluster for the
initial experiment. Do not mix MS-MPI and a Linux MPI runtime in one MPI job.
