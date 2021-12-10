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
#include "Hand.h"
using namespace cv;
using namespace std;

#define STAYING_STILL 0;
#define MOVE_DOWN 1;
#define MOVE_UP 2;
#define MOVE_LEFT 3;
#define MOVE_RIGHT 4;

Scalar const text_color = { 0, 255, 0 };
int const movement_threshold = 15;
int const local_skip_points = 5;

// FixComputedColor
// Precondition: Parameter is passed in correctly
// Postcondition: Will return num as an int and makes sure that
//                num will not be greater than 255 or less than 0.
int FixComputedColor(double num) {
	num = trunc(num);
	if (num > 255) {
		num = 255;
	}
	else if (num < 0) {
		num = 0;
	}
	return int(num);
}

// PrintHandLocation
// Precondition: Parameters are properly formatted and passed in correctly
// Postcondition: Will write the hand location on the passed in frame
void PrintHandLocation(Mat& frame, const Point hand_pos) {
	string hand_location = "Hand Location: (" + to_string(hand_pos.x) + ", " + to_string(hand_pos.y) + ")";
	putText(frame, hand_location, Point{ 3, frame.rows - 6 }, 1, 1.5, text_color, 2);
}

Mat MovementDirectionShape(const int direction) {
	Mat shape;
	if (direction == -1) {
		shape = imread("none.jpg");
	}
	else if (direction == 0) {
		shape = imread("stay.jpg");
	}
	else {
		shape = imread("arrow.jpg");
		if (direction == 1) {	//Down
			rotate(shape, shape, ROTATE_90_CLOCKWISE);
		}
		else if (direction == 2) {	//Up
			rotate(shape, shape, ROTATE_90_COUNTERCLOCKWISE);
		}
		else if (direction == 3) {	//Left
			rotate(shape, shape, ROTATE_180);
		}
		else {}	//Right
	}
	return shape;
}

// Puts text on the screen representing the hand type detected
// Preconditions: frame is of the correct type and correctly allocated, h_type is a constant integer
// Postconditions: A window with text representing the hand position matched is put on the screen
void PrintHandType(Mat& frame, const int h_type) {
	string type;

	if (h_type == -1) {
		type = "No Hand Detected";
	}
	else if (h_type == 0) {
		type = "Thumbs Up";
	}
	else if (h_type == 1) {
		type = "1 Finger Up";
	}
	else if (h_type == 2) {
		type = "2 Fingers Up";
	}
	else if (h_type == 3) {
		type = "3 Fingers Up";
	}
	else if (h_type == 4) {
		type = "4 Fingers Up";
	}
	else if (h_type == 5) {
		type = "5 Fingers Up";
	}

	string hand_type = "Hand Type: " + type;
	putText(frame, hand_type, Point{ 3, frame.rows - 30 }, 1, 1.5, text_color, 2);
}

// HandMovementDirection
// Precondition: Parameters are properly formatted and passed in correctly
// Postcondition: Will return an integer that tells which way the hand moved.
//                Either no moving/no hand detected, or up, down, left, or right
int HandMovementDirection(const Hand& current, const Hand& previous) {
	int change_in_x = current.location.x - previous.location.x;
	int change_in_y = current.location.y - previous.location.y;
	if (current.type == -1) {
		return -1;
	}
	if (change_in_x >= change_in_y) {
		if (change_in_x > movement_threshold && previous.type != -1) {
			if (change_in_x >= 0) {
				return MOVE_RIGHT;
			}
			else {
				return MOVE_LEFT;
			}
		}
		else {
			return STAYING_STILL;
		}
	}
	else {
		if (change_in_y > movement_threshold && previous.type != -1) {
			if (change_in_y >= 0) {
				return MOVE_DOWN;
			}
			else {
				return MOVE_UP;
			}
		}
		else {
			return STAYING_STILL;
		}
	}
}

// Finds the upper edge of the given object by finding which pixels have a value of 255 (white)
// Preconditions: object is of the correct type and is correctly allocated
// Postconditions: Returns a vector of points of the found top edges
vector<Point> FindTopEdge(const Mat& object) {
	vector<Point> points;
	for (int i = 0; i < object.cols; i += local_skip_points) {
		for (int j = 0; j < object.rows; j++) {
			if (object.at<uchar>(j, i) == 255) {//white
				points.push_back(Point(i, j));
				break;
			}
		}
	}
	return points;
}

int FindLocalMaximaMinima(const vector<Point>& points, const int middle) {
	vector<int> max, min;
	int smallest_value = 999999;
	int smallest_value_index = -1;
	int biggest_value = -1;
	int biggest_value_index = -1;

	for (int i = 1; i < points.size() - 1; i++) {
		//Finds the biggest and smallest value to find the middle
		if (points[i].y < smallest_value) {
			smallest_value = points[i].y;
			smallest_value_index = i;
		}
		if (points[i].y > biggest_value) {
			biggest_value = points[i].y;
			biggest_value_index = i;
		}

		bool skip = false;
		int next = i + 1;
		int prev = i - 1;
		if (points[next].y == points[i].y) {
			if ((next + 1) < int(points.size())) {
				next++;
				skip = true;
			}
			else break;
		}
		if (points[prev].y == points[i].y && (prev - 1) >= 0) {
			prev--;
			skip = true;
		}

		// Condition for local minima
		if ((points[prev].y > points[i].y) and
			(points[i].y < points[next].y))
			min.push_back(i);
		// Condition for local maxima
		else if ((points[prev].y < points[i].y) and
			(points[i].y > points[next].y))
			max.push_back(i);

		if (skip) {/////////////
			i++;
			skip = false;
		}
	}

	if (points[points.size() - 1].y == points[points.size() - 2].y) {
		if (points[points.size() - 2].y < points[points.size() - 3].y) {
			min.push_back(points.size() - 2);
		}
		else {
			max.push_back(points.size() - 2);
		}
	}
	if (points[0].y == points[1].y) {
		if (points[1].y < points[2].y) {
			min.push_back(1);
		}
		else {
			max.push_back(1);
		}
	}

	vector<int> true_minima;
	//int middle = ((biggest_value - smallest_value) / 2);
	for (int i = 0; i < min.size(); i++) {
		if (middle > points[min[i]].y) {
			true_minima.push_back(min[i]);
		}
	}
	vector<int> true_maxima;
	for (int i = 0; i < max.size(); i++) {
		if (middle > points[max[i]].y) {
			true_maxima.push_back(max[i]);
		}
	}

	//if (true_minima.size() - true_maxima.size() == 1 &&
	//if (true_minima.size() > true_maxima.size() &&
	if (true_minima.size() > 0 && true_maxima.size() < 6) {
		if (true_minima.size() == true_maxima.size()) {
			return (true_maxima.size() + 1);
		}
		else if (true_minima.size() - true_maxima.size() == 1) {
			return true_minima.size();
		}
	}
	return -1;
}
