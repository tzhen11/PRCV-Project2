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