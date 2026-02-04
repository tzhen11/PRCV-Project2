/*
    Name: Aafi Mansuri & Terry Zhen
*/

#include "distanceFunctions.h"

/*
    Computes euclidean distance between two features

    Parameters:
        a: feature vector 1
        b: feature vector 2

    Returns:
        euclidean distance between two features
*/
float euclideanDistance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        printf("Vector sizes do not match!");
        return -1;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

/*
    Computes histogram intersection between two histograms and normalizes
    the histograms and returns a similarity value where higher values indicate
    more similar histograms.
    
    To convert to distance metric, by subtracting intersection from 1, smaller values
    indicate more similar histograms.

    Parameters:
        a: histogram 1 (normalized)
        b: histogram 2 (normalized)

    Returns:
        histogram intersection distance (1 - intersection)
        -1 on error
*/
float histogramIntersection(const std::vector<float> &a, const std::vector<float> &b){
    if (a.size() != b.size()) {
        printf("Histogram sizes do not match!\n");
        return -1;
    }

    // Compute intersection: sum of minimum values at each bin
    float intersection = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        intersection += std::min(a[i], b[i]);
    }

    // Since intersection is in [0,1] where 1 equals identical image, return 1 - intersection so 0 = identical image now
    return 1 - intersection;
}