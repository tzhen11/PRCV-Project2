/*
	Name: Aafi Mansuri & Terry Zhen
	
	Purpose: Header file for image filter functions.
*/

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

// 3x3 Sobel X filter (detects vertical edges)
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

// 3x3 Sobel Y filter (detects horizontal edges)
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

// Gradient magnitude from Sobel X and Y
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
#endif