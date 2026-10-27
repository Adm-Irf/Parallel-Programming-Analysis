#pragma once
#include "matrix.hpp"
#include <string>
#include <vector>

// Each image is a 1 x 784 row; each label is a 1 x 10 one-hot row.
class MNISTLoader {
    std::vector<Matrix> trainingImages, trainingLabels, testImages, testLabels;
    static void readPair(const std::string& imagesPath, const std::string& labelsPath,
                         std::vector<Matrix>& images, std::vector<Matrix>& labels);
public:
    void loadDataset(const std::string& folder);
    const std::vector<Matrix>& getTrainingImages() const { return trainingImages; }
    const std::vector<Matrix>& getTrainingLabels() const { return trainingLabels; }
    const std::vector<Matrix>& getTestImages() const { return testImages; }
    const std::vector<Matrix>& getTestLabels() const { return testLabels; }
};
