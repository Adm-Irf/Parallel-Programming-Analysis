#include "mnist_loader.hpp"
#include <fstream>
#include <stdexcept>
#include <cstdint>

// IDX stores header integers with the most significant byte first (big-endian).
static uint32_t readNumber(std::ifstream& file) {
    unsigned char bytes[4];
    if (!file.read(reinterpret_cast<char*>(bytes), 4))
        throw std::runtime_error("Incomplete MNIST header.");
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

void MNISTLoader::readPair(const std::string& imagesPath, const std::string& labelsPath,
                          std::vector<Matrix>& images, std::vector<Matrix>& labels) {
    std::ifstream imageFile(imagesPath, std::ios::binary);
    std::ifstream labelFile(labelsPath, std::ios::binary);
    if (!imageFile || !labelFile) throw std::runtime_error("Cannot open MNIST files.");
    if (readNumber(imageFile) != 2051 || readNumber(labelFile) != 2049)
        throw std::runtime_error("Invalid MNIST file type.");
    uint32_t count = readNumber(imageFile);
    uint32_t labelCount = readNumber(labelFile);
    uint32_t rows = readNumber(imageFile), columns = readNumber(imageFile);
    if (count == 0 || count != labelCount || rows != 28 || columns != 28)
        throw std::runtime_error("MNIST images and labels must match and be 28 x 28.");

    images.clear();
    labels.clear();
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char pixels[784], digit;
        if (!imageFile.read(reinterpret_cast<char*>(pixels), 784) ||
            !labelFile.read(reinterpret_cast<char*>(&digit), 1) || digit > 9)
            throw std::runtime_error("Incomplete or invalid MNIST data.");

        // Normalize pixels to [0,1]; set the correct label's position to 1.
        Matrix image(1, 784, 0.0), label(1, 10, 0.0);
        for (size_t j = 0; j < 784; ++j) image.setElement(0, j, pixels[j] / 255.0);
        label.setElement(0, digit, 1.0);
        images.push_back(image);
        labels.push_back(label);
    }
}

void MNISTLoader::loadDataset(const std::string& folder) {
    readPair(folder + "/train-images.idx3-ubyte", folder + "/train-labels.idx1-ubyte",
             trainingImages, trainingLabels);
    readPair(folder + "/t10k-images.idx3-ubyte", folder + "/t10k-labels.idx1-ubyte",
             testImages, testLabels);
}
