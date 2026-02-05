/*
	Name: Aafi Mansuri & Terry Zhen

	Purpose: Image filter function implementations.
*/

#include "filters.h"

/*
	3x3 Sobel X filter - detects vertical edges.

	Apploes Sobel X filter (positive right) using the separable filters.
	Separable filters: [-1 0 1] horizontal, [1; 2; 1;] vertical
	Output is CV_16SC3 (signed short) since values can be negative.

	src: input color image
	dst: CV_16SC3 (signed short) sobel X image
*/
int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
	cv::Mat temp(src.size(), CV_16SC3);
	dst.create(src.size(), CV_16SC3);

	// Horizontal pass
	for (int i = 0; i < src.rows; i++) {
		cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i);
		cv::Vec3s* tempRow = temp.ptr<cv::Vec3s>(i);

		for (int j = 1; j < src.cols - 1; j++) {
			for (int c = 0; c < 3; c++) {
				// multiplying each term with -1, 0, 1 respectively
				tempRow[j][c] = -srcRow[j - 1][c] + srcRow[j + 1][c]; 
			}
		}
	}

	// Vertical pass
	for (int i = 1; i < src.rows - 1; i++) {
		cv::Vec3s* tempRowAbove = temp.ptr<cv::Vec3s>(i - 1);
		cv::Vec3s* tempRow = temp.ptr<cv::Vec3s>(i);
		cv::Vec3s* tempRowBelow = temp.ptr<cv::Vec3s>(i + 1);
		cv::Vec3s* dstRow = dst.ptr<cv::Vec3s>(i);

		for (int j = 1; j < src.cols - 1; j++) {
			for (int c = 0; c < 3; c++) {
				// multiplying each term with 1, 2, 1 respectively
				dstRow[j][c] = tempRowAbove[j][c] + 2 * tempRow[j][c] + tempRowBelow[j][c]; 
			}
		}
	}

	return 0;
}

/*
	3x3 Sobel Y filter - detects horizontal edges.

	Apploes Sobel Y filter (positive up) using the separable filters.
	Separable filters: [1 2 1] horizontal, [1; 0; -1;] vertical
	Output is CV_16SC3 (signed short) since values can be negative.

	src: input color image
	dst: CV_16SC3 (signed short) sobel Y image
*/
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
	cv::Mat temp(src.size(), CV_16SC3);
	dst.create(src.size(), CV_16SC3);

	// Horizontal pass
	for (int i = 0; i < src.rows; i++) {
		cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i);
		cv::Vec3s* tempRow = temp.ptr<cv::Vec3s>(i);

		for (int j = 1; j < src.cols - 1; j++) {
			for (int c = 0; c < 3; c++) {
				// multiplying each term with 1, 2, 1 respectively
				tempRow[j][c] = srcRow[j - 1][c] + 2 * srcRow[j][c] + srcRow[j + 1][c];
			}
		}
	}

	// Vertical pass
	for (int i = 1; i < src.rows - 1; i++) {
		cv::Vec3s* tempRowAbove = temp.ptr<cv::Vec3s>(i - 1);
		cv::Vec3s* tempRowBelow = temp.ptr<cv::Vec3s>(i + 1);
		cv::Vec3s* dstRow = dst.ptr<cv::Vec3s>(i);

		for (int j = 1; j < src.cols - 1; j++) {
			for (int c = 0; c < 3; c++) {
				// multiplying each term with 1, 0, -1 respectively
				dstRow[j][c] = tempRowAbove[j][c] + -tempRowBelow[j][c];
			}
		}
	}

	return 0;
}

/*
	Gradient magnitude from Sobel X and Y

	Computes gradient magnitude image from the Sobel X and Sobel Y images.
	Uses Euclidean distance sqrt(sx^2 + sy^2)

	sx:  CV_16SC3 (signed short) sobel X image
	sy:	 CV_16SC3 (signed short) sobel Y image
	dst: CV_8UC3 output image.
*/
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
	dst.create(sx.size(), CV_8UC3);

	for (int i = 0; i < sx.rows; i++) {
		cv::Vec3s* sxRow = sx.ptr<cv::Vec3s>(i);
		cv::Vec3s* syRow = sy.ptr<cv::Vec3s>(i);
		cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(i);

		for (int j = 0; j < sx.cols; j++) {
			for (int c = 0; c < 3; c++) {
				double mag = sqrt(sxRow[j][c] * sxRow[j][c] + syRow[j][c] * syRow[j][c]);
				dstRow[j][c] = cv::saturate_cast<uchar>(mag); // Clip values to [0,255]
			}
		}
	}

	return 0;
}