/*
	Name: Aafi Mansuri & Terry Zhen

	Purpose: The first program is given a directory of images and
    feature set and it writes the feature vector for each image to a file.
*/	

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include "csv_util.h"
#include "featureMethods.h"

// Define filesystem
namespace fs = std::filesystem;

/*
    Retrieve image files from directory

    Parameters:
        directory: path to directory of images
        imageFiles: output vector of image file paths
    
    Returns:
        number of images found
        -1, on error
*/

int retrieveImageFiles(const std::string &directory, std::vector<std::string> &imageFiles) {
    imageFiles.clear();

    // Check if directory exists
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        printf("Error! Directory does not exist!\n");
        return -1;
    }

    for (const auto &entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        fs::path filepath = entry.path();
        std::string ext = filepath.extension().string();

        if (ext == ".jpg") {
            imageFiles.push_back(filepath.string());
        }
    }

    // Return number of files
    return static_cast<int>(imageFiles.size());
}

// Generate features in csv for image matching
int main(int argc, char* argv[]) {
    // Parse arguments
    std::string dbDirectory = argv[1];
    std::string featureMethod = argv[2];
    char* outputCSV = argv[3];

    std::vector<std::string> imageFiles;
    int numImages = retrieveImageFiles(dbDirectory, imageFiles);

    printf("Found %d images.\n", numImages);

    // Control for wiping csv and appending
    int reset = 1;

    // Extract feature vector from each image
    for (const auto &imgPath: imageFiles) {
        // Read image
        cv::Mat image = cv::imread(imgPath);
        if (image.empty()) {
            continue;
        }
        
        std::vector<float> features;
        int status = -1;
        // Run baseline feature method
        if (featureMethod == "baseline") {
            status = baseline7x7(image, features);
        }
        else if (featureMethod == "chistogram") {
            status = colorHistogram(image, features);
        }
        else {
            printf("Error, feature method not valid!\n");
            return -1;
        }

        if (status != 0) {
            printf("Warning: Feature extraction failed for %s\n", imgPath.c_str());
            continue;
        }

        append_image_data_csv(outputCSV, const_cast<char*>(imgPath.c_str()), features, reset);

        reset = 0;       
    }

    return 0;
}