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
int colorHistogram(const cv::Mat &src, std::vector<float> &features, int histSize = 16);

/*
    Function to compute multi-region 2D RG chromaticity histograms.
    Computes two histograms: whole image and center region.
    
    Parameters:
        src: input image (BGR format)
        features: output feature vector (size = 2 * histSize * histSize)
                  [whole_hist_flattened, center_hist_flattened]
        histSize: number of bins per dimension (default 16)
    
    Returns:
        0 on success
        -1 on error
*/
int multiHistogram(const cv::Mat &src, std::vector<float> &features, int histSize = 16);

/*
    Function to compute combined texture and color features from an image.
    
    Color: 2D RG chromaticity histogram (same as colorHistogram function)
    Texture: Histogram of gradient magnitudes computed from Sobel X and Y filters
    
    The final feature vector concatenates both histograms:
    [color histogram (histSize * histSize values)] + [texture histogram (histSize values)]
    
    Parameters:
        src: input image (BGR format)
        features: output feature vector (flattened, size = histSize * histSize + histSize)
        histSize: number of bins per dimension (default 16)
    
    Returns:
        0 on success
        -1 on error
*/
int textureAndColor(const cv::Mat &src, std::vector<float> &features, int histSize = 16);


/*
    Function to compute face-aware multi-region histograms.
    Requires face detection. Returns error if no face found.
    Computes whole image, face region, and background histograms.
    
    Parameters:
        src: input image (BGR format)
        features: output feature vector (3 * histSize * histSize)
        histSize: number of bins per dimension (default 16)
    
    Returns:
        0 on success
        -1 on error/no face found
*/
int faceDetectHistogram(const cv::Mat &src, std::vector<float> &features, int histSize = 16);

#endif