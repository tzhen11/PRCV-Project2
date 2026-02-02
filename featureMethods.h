/*
    Name: Aafi Mansuri & Terry Zhen
*/

#ifndef FEATUREMETHODS_H
#define FEATUREMETHODS_H

#include "opencv2/opencv.hpp"

/*
    Function to extract center 7x7 square from image as feature vector.

    Parameters:
        src: input image
        features: output feature vector (7*7 = 49 values for grayscale, 7*7*3 = 147 values for BGR)
    
    Returns:
        0 on success
        -1 on error
*/
int baseline7x7(const cv::Mat &src, std::vector<float> &features);

#endif