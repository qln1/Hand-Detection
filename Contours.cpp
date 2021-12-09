// Author: Quintin Nguyen, Akhil Lal, Matthew Cho

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <cmath>
#include <opencv2/core/types.hpp>
#include <vector>
#include <stdlib.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include "Helper.cpp"
#include "ImageOperations.cpp"
#include "ObjectRecognition.cpp"
using namespace cv;
using namespace std;

#define STAYING_STILL 0;
#define MOVE_DOWN 1;
#define MOVE_UP 2;
#define MOVE_LEFT 3;
#define MOVE_RIGHT 4;

Scalar const text_color = { 0, 255, 0 };
string const video_name_path = "hand.mp4";
double const contrast_num = 1.25;
int const brightness_level = 12;/////////////////////
int const local_skip_points = 5;
int const gaus_blur_size = 11;
int const gaus_blur_amount = 3;
int const background_remover_thresh = 18;
int const median_blur = 7;
string const template_path = "Templates\\hand";
int const skip_frames = 3;/////////////////////////////////////////////
int const min_contour_area = 8000;
int const similarity_threshold = 10;////////////////////////////////////
int const movement_threshold = 15;
int const min_hessian = 400;
float const ratio_thresh = 0.7;
Scalar const box_color = Scalar(0, 0, 255);
int const number_random_frames = 80;/////////////////////////////////
int const sat_val = 15;

int test = 0;

// Finds the image contours in the given image and puts them in a vector, and returns it
// Preconditions: image is of the correct type and correctly allocated
// Postconditions: vector of contours within the image is returned
vector<vector<Point>> FindImageContours(const Mat& object) {
	Mat thresh;
	threshold(object, thresh, 90, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	return contours;
}

// Finds the nth biggest contour in the given list of contours, biggest is determined by rectangular area of the contour
// Preconditions: contours list and box is of the correct type and are correctly allocated, n is an constant integer
// Postconditions: Returns the index of the nth biggest contour
int FindNthBiggestContour(const vector<vector<Point>>& contours, Rect& box, const int n) {
	int index = contours.size() - n;
	if (contourArea(contours[index]) >= min_contour_area) {
		box = boundingRect(contours[index]);
		return index;
	}
	return -1;
}

// Determines which of the given contours is bigger
// Preconditions: contour1 and contour2 are of the correct type and are correctly allocated
// Postconditions: Returns true if contour2 is bigger than contour1, false if not
bool CompareContourAreas(const vector<Point> contour1, const vector<Point> contour2) {///////////////////////////////////////////////
	double i = fabs(contourArea(Mat(contour1)));
	double j = fabs(contourArea(Mat(contour2)));
	return (i < j);
}