/*
    Name: Aafi Mansuri & Terry Zhen
*/

#include "featureMethods.h"
#include "filters.h"

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
int textureAndColor(const cv::Mat &src, std::vector<float> &features, int histSize) {
    features.clear();

    // Validate input
    if (src.empty()) {
        printf("Error, source is empty!\n");
        return -1;
    }

    if (src.channels() != 3) {
        printf("Error, image must be 3-channel!\n");
        return -1;
    }

    // Color Histogram
    std::vector<float> colorFeatures;
    int status = colorHistogram(src, colorFeatures, histSize);
    if (status != 0) {
        printf("Error computing color histogram!\n");
        return -1;
    }

    //Texture Histogram
    
    // Convert to grayscale for texture analysis
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Compute Sobel X and Y using our own functions
    cv::Mat sobelX, sobelY;
    cv::Mat srcCopy = src.clone();
    sobelX3x3(srcCopy, sobelX);
    sobelY3x3(srcCopy, sobelY);

    // Compute gradient magnitude (outputs CV_8UC3)
    cv::Mat mag;
    magnitude(sobelX, sobelY, mag);

    // Convert color magnitude to grayscale by averaging channels
    cv::Mat grayMag;
    cv::cvtColor(mag, grayMag, cv::COLOR_BGR2GRAY);

    // Build texture histogram from magnitudes
    // Magnitude values are in [0, 255], map to bins
    cv::Mat textureHist = cv::Mat::zeros(1, histSize, CV_32FC1);

    float maxLog = std::log(256.0f);

    for (int i = 0; i < grayMag.rows; i++) {
        const uchar* magPtr = grayMag.ptr<uchar>(i);
        for (int j = 0; j < grayMag.cols; j++) {
            // Log scale: spreads out low values
            float logMag = std::log(1.0f + magPtr[j]);
            
            // Map to bin index
            int binIndex = static_cast<int>((logMag / maxLog) * (histSize - 1) + 0.5f);
            binIndex = std::min(std::max(binIndex, 0), histSize - 1);
            textureHist.at<float>(0, binIndex)++;
        }
    }
    // Normalize texture histogram
    int totalPixels = grayMag.rows * grayMag.cols;
    textureHist /= totalPixels;
   
    // Add color features
    for (float val : colorFeatures) {
        features.push_back(val);
    }

    // Add texture features
    for (int i = 0; i < histSize; i++) {
        features.push_back(textureHist.at<float>(0, i));
    }

    return 0;
}
