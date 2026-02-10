/*
    Name: Aafi Mansuri & Terry Zhen
*/

#include "distanceFunctions.h"
#include <cstdio>   
#include <cmath>

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

/*
    Computes distance for multi-histogram features.
    Splits the feature vector into two histograms, compares each,
    then combines using weighted average.
    
    Parameters:
        a: multi-histogram feature vector 1 (size = 2 * histSize * histSize)
        b: multi-histogram feature vector 2 (size = 2 * histSize * histSize)
        wholeWeight: weight for whole image histogram (default 0.5), centerWeight = 1.0 - weightWhole
    
    Returns:
        combined distance
        -1 on error
*/
float multiHistogramDistance(const std::vector<float> &a, const std::vector<float> &b, float wholeWeight) {
    // Validate histograms
    if (a.size() != b.size()) {
        printf("Histogram sizes do not match!\n");
        return -1;
    }

    int halfSize = a.size() / 2;

    // Split into whole and center histograms
    std::vector<float> aWhole(a.begin(), a.begin() + halfSize);
    std::vector<float> aCenter(a.begin() + halfSize, a.end());

    std::vector<float> bWhole(b.begin(), b.begin() + halfSize);
    std::vector<float> bCenter(b.begin() + halfSize, b.end());

    // Compute distances for each histogram
    float wholeDist = histogramIntersection(aWhole, bWhole);
    float centerDist = histogramIntersection(aCenter, bCenter);

    // Incorprate weight
    float centerWeight = 1.0f - wholeWeight;
    float combinedDist = wholeWeight * wholeDist + centerWeight * centerDist;

    return combinedDist;
}

/*
    Computes weighted distance combining color and texture histograms.
    
    Separates the combined feature vector into color and texture portions,
    computes histogram intersection for each, converts to distance,
    and returns a weighted combination.

    Parameters:
        a: combined feature vector 1 [color (histSize*histSize) + texture (histSize)]
        b: combined feature vector 2 [color (histSize*histSize) + texture (histSize)]
        colorWeight: weight for color distance (default 0.5, range 0-1)
        histSize: number of bins per histogram dimension (default 16)

    Returns:
        weighted distance where 0 = identical, 1 = completely different
        -1 on error
*/
float textureColorDistance(const std::vector<float> &a, const std::vector<float> &b, 
                           float colorWeight, int histSize) {
    int colorSize = histSize * histSize;
    
    if (a.size() != b.size()) {
        printf("Feature vector size mismatch!\n");
        return -1;
    }

    // Color histogram intersection (first 256 values)
    float colorIntersection = 0.0f;
    for (int i = 0; i < colorSize; i++) {
        colorIntersection += std::min(a[i], b[i]);
    }

    // Texture histogram intersection (last 16 values)
    float textureIntersection = 0.0f;
    for (size_t i = colorSize; i < a.size(); i++) {
        textureIntersection += std::min(a[i], b[i]);
    }

    // Convert to distances (each intersection is 0-1 range)
    float colorDist = 1.0f - colorIntersection;
    float textureDist = 1.0f - textureIntersection;

    // Weighted combination
    float textureWeight = 1.0f - colorWeight;
    return colorWeight * colorDist + textureWeight * textureDist;
}

/*
    Computes distance for face-detect features.
    Assumes both feature vectors are from images that have face(s) (768 features).
    
    Parameters:
        a: face-aware feature vector 1 (size = 3 * histSize * histSize)
        b: face-aware feature vector 2 (size = 3 * histSize * histSize)
        wholeWeight: weight for whole histogram (default 0.2)
        faceWeight: weight for face histogram (default 0.6)
        backgroundWeight: weight for background histogram (default 0.2)
        histSize: each histogram size (default 16)
    
    Returns:
        combined distance
        -1 on error
*/
float faceDetectDistance(const std::vector<float> &a, const std::vector<float> &b,
                        float wholeWeight, float faceWeight, float backgroundWeight, int histSize) {
    // Check vector sizes
    if (a.size() != b.size()) {
        printf("Feature vector sizes don't match!\n");
        return -1;
    }

    int oneHistogramSize = histSize * histSize;

    // Separate the three histograms
    std::vector<float> aWhole(a.begin(), a.begin() + oneHistogramSize);
    std::vector<float> aFace(a.begin() + oneHistogramSize, a.begin() + 2 * oneHistogramSize);
    std::vector<float> aBackground(a.begin() + 2 * oneHistogramSize, a.end());

    std::vector<float> bWhole(b.begin(), b.begin() + oneHistogramSize);
    std::vector<float> bFace(b.begin() + oneHistogramSize, b.begin() + 2 * oneHistogramSize);
    std::vector<float> bBackground(b.begin() + 2 * oneHistogramSize, b.end());

    // Compute distance for each histogram
    float wholeDist = histogramIntersection(aWhole, bWhole);
    float faceDist = histogramIntersection(aFace, bFace);
    float backgroundDist = histogramIntersection(aBackground, bBackground);

    // Weighted distance of all three
    float combinedDist = wholeWeight * wholeDist + faceWeight * faceDist + backgroundWeight * backgroundDist;

    return combinedDist;
}

/*
    Computes cosine distance between two feature vectors.
    
    Cosine similarity = (a · b) / (||a|| × ||b||)
    Cosine distance = 1 - similarity
    
    Parameters:
        a: feature vector 1
        b: feature vector 2
    
    Returns:
        cosine distance (0 = identical, 2 = opposite)
*/
float cosineDistance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        printf("Vector sizes do not match!\n");
        return -1;
    }

    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    for (size_t i = 0; i < a.size(); i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    normA = std::sqrt(normA);
    normB = std::sqrt(normB);

    if (normA == 0 || normB == 0) return 1.0f;

    float similarity = dotProduct / (normA * normB);
    return 1.0f - similarity;
}