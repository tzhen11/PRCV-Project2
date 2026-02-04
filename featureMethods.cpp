/*
    Name: Aafi Mansuri & Terry Zhen
*/

#include "featureMethods.h"

/*
    Function to extract center 7x7 square from image as feature vector.

    Parameters:
        src: input image
        features: output feature vector (7*7 = 49 values for grayscale, 7*7*3 = 147 values for BGR)
    
    Returns:
        0 on success
        -1 on error
*/
int baseline7x7(const cv::Mat &src, std::vector<float> &features) {
    features.clear();

    // Validate image size
    if (src.rows < 7 || src.cols < 7) {
        printf("Image too small, less than 7x7!");
        return -1;
    }

    // Compute center position
    int centerRow = src.rows / 2;
    int centerCol = src.cols / 2;

    // Define starting position of 7x7 (top left corner of square)
    int startRow = centerRow - 3;
    int startCol = centerCol - 3;

    // Extract 7x7 square region
    cv::Mat centerSquare = src(cv::Rect(startCol, startRow, 7, 7));

    // Flatten 7x7 square into feature vector
    for (int i = 0; i < centerSquare.rows; i++) {
        for (int j = 0; j < centerSquare.cols; j++) {
            if (centerSquare.channels() == 1) {
                // Add to feature vector
                features.push_back(static_cast<float>(centerSquare.at<uchar>(i, j)));
            }
            else if (centerSquare.channels() == 3) {
                // Add each channel to feature vector
                cv::Vec3b pixel = centerSquare.at<cv::Vec3b>(i, j);
                features.push_back(static_cast<float>(pixel[0])); // B
                features.push_back(static_cast<float>(pixel[1])); // G
                features.push_back(static_cast<float>(pixel[2])); // R
            }
        }
    }

    return 0;
}

/*
    Function to compute a 2D RG chromaticity histogram from an image.
    The histogram uses r and g chromaticity values with normalized RBG.
    
    Parameters:
        src: input image (BGR format)
        features: output feature vector (flattened histogram, size = histSize * histSize)
        histSize: number of bins per dimension (default 16)
    
    Returns:
        0 on success
        -1 on error
*/
int colorHistogram(const cv::Mat &src, std::vector<float> &features, int histSize) {
    features.clear();

    // Validate the inputs
    if (src.empty()) {
        printf("Error, source is empty!\n");
        return -1;
    }

    if (src.channels() != 3) {
        printf("Error, image must be 3-channel!\n");
        return -1;
    }

    // Initialize histogram with zeros
    cv::Mat hist = cv::Mat::zeros(cv::Size(histSize, histSize), CV_32FC1);

    // Compute histogram, loop over all pixels
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);
        
        for (int j = 0; j < src.cols; j++) {
            // Retrieve RGB vals
            float B = srcPtr[j][0];
            float G = srcPtr[j][1];
            float R = srcPtr[j][2];

            // Compute chromaticity vals (r , g)
            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0; // check for all zeros (avoid divide by 0)
            float r = R / divisor;
            float g = G / divisor;

            // Compute histogram bin indicies (r and g are in [0, 1] range, map to [0, histogram size - 1])
            int rIndex = static_cast<int>(r * (histSize - 1) + 0.5);
            int gIndex = static_cast<int>(g * (histSize - 1) + 0.5);

            // Clamp indicies to valid range
            rIndex = std::min(std::max(rIndex, 0), histSize - 1);
            gIndex = std::min(std::max(gIndex, 0), histSize - 1);

            // Increment histogram bin
            hist.at<float>(rIndex, gIndex)++;
        }
    }

    // Normalize RBG
    int totalPixels = src.rows * src.cols;
    hist /= totalPixels;

    // Flatten histogram for feature vector
    for (int i = 0; i < hist.rows; i++) {
        const float *histPtr = hist.ptr<float>(i);
        for (int j = 0; j < hist.cols; j++) {
            features.push_back(histPtr[j]);
        }
    }
    assert(features.size() == histSize * histSize);
    
    return 0;
}