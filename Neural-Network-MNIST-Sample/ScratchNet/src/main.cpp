#include "neural_network.hpp"
#include "mnist_loader.hpp"
#include "parallel.hpp"
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <stdexcept>

// Fixed settings keep the sequential and parallel experiments comparable.
constexpr int BATCH_SIZE = 64;
constexpr float LEARNING_RATE = 0.01f;
constexpr unsigned int SEED = 42;

int positiveInteger(const char* text) {
    size_t used = 0;
    int value = std::stoi(text, &used);
    if (used != std::string(text).size() || value < 1)
        throw std::invalid_argument("Epochs and threads must be positive integers.");
    return value;
}

int main(int argc, char* argv[]) {
    // Initializes MPI only in the MPI build. Rank 0 prints the results.
    parallel::Runtime runtime(argc, argv);
    if (parallel::rank != 0) std::cout.setstate(std::ios::failbit);
    try {
        if (argc > 4) throw std::invalid_argument("Usage: program [epochs] [threads] [data_folder]");
        const int epochs = argc > 1 ? positiveInteger(argv[1]) : 5;
        const int threads = argc > 2 ? positiveInteger(argv[2]) : 1;
        const std::string dataFolder = argc > 3 ? argv[3] : "../../MNIST";

#ifdef _OPENMP
        // Request the CPU threads used by the matrix multiplication loops.
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
#pragma omp parallel
        {
#pragma omp single
            parallel::threads = omp_get_num_threads();
        }
#else
        if (threads != 1) throw std::invalid_argument("Only the OpenMP version accepts threads > 1.");
#endif

        // All versions read the same four IDX files.
        for (const char* file : {"train-images.idx3-ubyte", "train-labels.idx1-ubyte",
                                "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"}) {
            if (!std::filesystem::is_regular_file(std::filesystem::path(dataFolder) / file))
                throw std::runtime_error("Missing file: " + dataFolder + "/" + file);
        }
        MNISTLoader loader;
        loader.loadDataset(dataFolder);
        const auto& images = loader.getTrainingImages();
        const auto& labels = loader.getTrainingLabels();
        const auto& testImages = loader.getTestImages();
        const auto& testLabels = loader.getTestLabels();

        // Create the same starting model for every experiment.
        Layer::setSeed(SEED);
        NeuralNetwork network;
        network.setLossFunction("cross_entropy");
        network.addLayer(Layer(784, 128, "relu"));
        network.addLayer(Layer(128, 64, "relu"));
        network.addLayer(Layer(64, 10, "softmax"));

        // MPI sends rank 0's initial weights to every laptop.
        for (size_t i = 0; i < network.getLayerCount(); ++i) {
            Matrix weights = network.getLayer(i).getWeights();
            Matrix biases = network.getLayer(i).getBiases();
            parallel::exchange(weights, true);
            parallel::exchange(biases, true);
            network.getLayer(i).setWeights(weights);
            network.getLayer(i).setBiases(biases);
        }

        std::cout << "\nConfiguration\n"
                  << "Mode: " << parallel::mode() << " | Threads: " << parallel::threads
                  << " | Processes: " << parallel::size << '\n'
                  << "Training images: " << images.size() << " | Test images: " << testImages.size() << '\n'
                  << "Network: 784 -> 128 -> 64 -> 10\n"
                  << "Epochs: " << epochs << " | Batch: " << BATCH_SIZE
                  << " | Learning rate: " << LEARNING_RATE << " | Seed: " << SEED << '\n';

        // Time training only: loading, initialization and testing are excluded.
        parallel::barrier();
        auto start = std::chrono::steady_clock::now();
        network.train(images, labels, epochs, LEARNING_RATE, BATCH_SIZE, SEED);
        double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        // Distributed training finishes when the slowest process finishes.
        seconds = parallel::maximum(seconds);

        // Test once on rank 0. The largest output identifies the predicted digit.
        if (parallel::rank == 0) {
            size_t correct = 0;
            for (size_t i = 0; i < testImages.size(); ++i) {
                Matrix output = network.predict(testImages[i]);
                size_t predicted = 0, actual = 0;
                for (size_t digit = 0; digit < 10; ++digit) {
                    if (!std::isfinite(output.getElement(0, digit)))
                        throw std::runtime_error("Training produced a non-finite prediction.");
                    if (output.getElement(0, digit) > output.getElement(0, predicted)) predicted = digit;
                    if (testLabels[i].getElement(0, digit) > 0.5) actual = digit;
                }
                if (predicted == actual) ++correct;
            }
            std::cout << std::fixed << std::setprecision(3)
                      << "\nResults\nTraining time: " << seconds << " seconds\n"
                      << "Correct predictions: " << correct << "/" << testImages.size() << '\n'
                      << "Accuracy: " << 100.0 * correct / testImages.size() << "%\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
#ifdef USE_MPI
        // Stop all ranks if one fails, rather than leaving others waiting.
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
}
