/*
	Name: Aafi Mansuri & Terry Zhen

	Purpose: The second program is given a target image,
    the feature set, and the feature vector file. It then computes
    the features for the target image, reads the feature vector file,
    and identifies the top N matches.
*/	

#include <iostream>
#include <string>
#include "csv_util.h"
#include "featureMethods.h"
#include "distanceFunctions.h"

// Computes top N matches from image DB to target image using euclidean distance
int main(int argc, char* argv[]) {
    // Parse arguments
    char* targetImagePath = argv[1];
    std::string featureMethod = argv[2];
    char* featureCSV = argv[3];
    int N = std::atoi(argv[4]);

    // Read feature CSV
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;
    read_image_data_csv(featureCSV, filenames, data, 0);

    // Load target image
    cv::Mat targetImage = cv::imread(targetImagePath);
    if (targetImage.empty()){
        printf("Error loading target image!\n");
        return -1;
    }

    // Target features
    std::vector<float> targetFeatures;
    int status = -1;

    if (featureMethod == "resnet") {
        // Lookup from CSV for ResNet
        std::string targetName = targetImagePath;
        
        // Extract filename only (remove path)
        size_t pos = targetName.find_last_of("/\\");
        if (pos != std::string::npos) {
            targetName = targetName.substr(pos + 1);
        }
        
        // Find in CSV
        for (size_t i = 0; i < filenames.size(); i++) {
            if (std::string(filenames[i]) == targetName) {
                targetFeatures = data[i];
                status = 0;
                break;
            }
        }
        
        if (status != 0) {
            printf("Error: Target image not found in ResNet CSV!\n");
            return -1;
        }
    }
    else {
        // Compute features for tasks 1-4
        cv::Mat targetImage = cv::imread(targetImagePath);
        if (targetImage.empty()) {
            printf("Error loading target image!\n");
            return -1;
        }

        if (featureMethod == "baseline") {
            status = baseline7x7(targetImage, targetFeatures);
        }
        else if (featureMethod == "chistogram") {
            status = colorHistogram(targetImage, targetFeatures, 16);
        }
        else if (featureMethod == "texture") {
            status = textureAndColor(targetImage, targetFeatures);
        }
        else {
            printf("Feature method not valid!\n");
            return -1;
        }

        if (status != 0) {
            printf("Error: Feature extraction failed!\n");
            return -1;
        }
    }


    // Compute distances
    std::vector<std::pair<float, std::string>> results;

    for (size_t i = 0; i < data.size(); i++) {
        float dist = -1.0f;

        // Compute distance using appropriate distance metric
        if (featureMethod == "baseline") {
            dist = euclideanDistance(targetFeatures, data[i]);
        }
        else if (featureMethod == "chistogram") {
            dist = histogramIntersection(targetFeatures, data[i]);
        }
        else if (featureMethod == "mhistogram") {
            dist = multiHistogramDistance(targetFeatures, data[i], 0.5f);
        }
        else if (featureMethod == "texture") {
            dist = textureColorDistance(targetFeatures, data[i],0.3f);
        }
        else if (featureMethod == "resnet") {
            dist = euclideanDistance(targetFeatures, data[i]);
        }
        // Store results
        if (dist >= 0) {
            results.emplace_back(dist, filenames[i]);
        }
    }

    // Sort by ascending distance
    std::sort(results.begin(), results.end(),
                [](const auto &a, const auto &b) {
                    return a.first < b.first;
                });
    
    // Output top N image matches
    printf("The top %d image matches:\n", N);

    for (int i = 0; i < std::min(N, (int)results.size()); i++) {
        printf("%d: %s  (distance = %.5f)\n", i + 1,results[i].second.c_str(), results[i].first);
    }

    // Cleanup
    for (char* f : filenames) {
        delete[] f;
    }

    return 0;
}