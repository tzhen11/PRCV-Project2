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

    // Load target image
    cv::Mat targetImage = cv::imread(targetImagePath);
    if (targetImage.empty()){
        printf("Error loading target image!\n");
        return -1;
    }

    // Compute target features
    std::vector<float> targetFeatures;
    int status = -1;
    if (featureMethod == "baseline") {
        status = baseline7x7(targetImage, targetFeatures);
    }
    else if (featureMethod == "chistogram") {
        status = colorHistogram(targetImage, targetFeatures, 16);
    }
    else if (featureMethod == "mhistogram") {
        status = multiHistogram(targetImage, targetFeatures, 16);
    }
    else if (featureMethod == "texture") {
        status = textureAndColor(targetImage, targetFeatures);
    }
    else {
        printf("Feature function not valid!\n");
        return -1;
    }

    if (status != 0) {
        printf("Error, feature extraction failed for target image!\n");
        return -1;
    }

    // Read feature CSV
    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;

    read_image_data_csv(featureCSV, filenames, data, 0);

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

    // Cleanup allocated filenames
    for (char* f : filenames) {
        delete[] f;
    }

    return 0;
}