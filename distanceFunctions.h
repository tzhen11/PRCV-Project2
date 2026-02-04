/*
    Name: Aafi Mansuri & Terry Zhen
*/
#ifndef DISTANCEFUNCTIONS_H
#define DISTANCEFUNCTIONS_H

#include <vector>

/*
    Computes euclidean distance between two features

    Parameters:
        a: feature vector 1
        b: feature vector 2

    Returns:
        euclidean distance between two features
*/
float euclideanDistance(const std::vector<float> &a, const std::vector<float> &b);

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
float histogramIntersection(const std::vector<float> &a, const std::vector<float> &b);

#endif