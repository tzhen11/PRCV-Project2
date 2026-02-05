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

/*
    Computes weighted distance combining color and texture histograms.
    
    Separates the combined feature vector into color and texture portions,
    computes histogram intersection for each, converts to distance,
    and returns a weighted combination.

    Parameters:
        a: combined feature vector [color (histSize*histSize) + texture (histSize)]
        b: combined feature vector [color (histSize*histSize) + texture (histSize)]
        colorWeight: weight for color distance (default 0.5, range 0-1)
        histSize: number of bins per histogram dimension (default 16)

    Returns:
        weighted distance where 0 = identical, 1 = completely different
        -1 on error
*/
float textureColorDistance(const std::vector<float> &a, const std::vector<float> &b, 
                           float colorWeight = 0.5f, int histSize = 16);
#endif