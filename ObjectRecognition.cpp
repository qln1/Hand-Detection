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

// HowSimilarImagesAre
// Preconditions: a valid hand template and front image containing a hand are passed in. This program should also
// contain and use OpenCV files in order to use SIFT.
// Postconditions: An image showing the keypoint matches between the hand template and front image is shown.
// The number of keypoints in the vector good_matches is returned, indicating how strong the match was.
// Note: waits for a key press between each image display
int HowSimilarImagesAre(const Mat& hand, const Mat& front) {
	Ptr<SIFT> detector = SIFT::create(min_hessian);
	vector<KeyPoint> keypoints_template, keypoints_search;
	Mat descriptor_template, descriptor_search;
	detector->detectAndCompute(hand, noArray(), keypoints_template, descriptor_template);
	imshow("ssss", front);
	waitKey(0);

	detector->detectAndCompute(front, noArray(), keypoints_search, descriptor_search);

	Ptr<DescriptorMatcher> feature_matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	vector< vector<DMatch> > matches;
	feature_matcher->knnMatch(descriptor_template, descriptor_search, matches, 2);

	vector<DMatch> good_matches;
	for (size_t i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
			good_matches.push_back(matches[i][0]);
		}
	}

	Mat drawn_matches;
	drawMatches(hand, keypoints_template, front, keypoints_search, good_matches, drawn_matches, Scalar::all(-1),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Good Matches", drawn_matches);
	waitKey();

	return (int)good_matches.size();
}

// TemplateMatchingWithObject
// Preconditions: Numbered template files exist in the project directory and the function HowSimilarImagesAre
// is fully functional and implemented.
// Postconditions: Either returns an integer that matches a template with the hand position in the front image
// passed in or returns -1, signifying that a hand was failed to detect.
int TemplateMatchingWithObject(const Mat& front) {
	//Possible detect left vs right here first???

	int most_similar_file = -1;
	int highest_similar_value = -1;

	int i = 0;
	while (true) {
		string file_name = template_path + to_string(i) + ".jpg";
		Mat hand = imread(file_name);

		if (hand.empty()) break;
		int current_simi = HowSimilarImagesAre(hand, front);
		cout << i << " " << current_simi << endl;
		if (current_simi > highest_similar_value) {
			highest_similar_value = current_simi;
			most_similar_file = i;
		}
		i++;
	}
	if (highest_similar_value >= similarity_threshold) {	//matches a template
		return most_similar_file;
	}
	else {	//fails to detect hand
		return -1;
	}
}

// SearchForHand
// Preconditions: The functions FindNthBiggestContour and FindLocalMaximaMinima exist and are fully implemented.
// Postconditions: A hand class is returned with the following values: the type and the x and y location coordinates.
Hand SearchForHand(const Mat& front, const vector<vector<Point>>& contours, Rect& box) {
	Hand hand;
	Mat only_object;
	for (int i = 1; i <= contours.size(); i++) {
		int contour_index = FindNthBiggestContour(contours, box, i);
		if (contour_index == -1) {
			break;
		}
		Mat pic(front.rows, front.cols, CV_8U, Scalar::all(0));
		drawContours(pic, contours, contour_index, Scalar(255, 255, 255), FILLED);

		only_object = pic(box);

		//imshow("dfdfsd", only_object);
		//waitKey(0);
		//imwrite(to_string(test) + "pppppppppppppppppppppppppppppppppppppp.jpg", only_object);
		//test++;

		int type = FindLocalMaximaMinima(FindTopEdge(only_object), (only_object.rows / 2));

		//int type = TemplateMatchingWithObject(only_object);
		//cout << "TYPEE!! " << type << endl;
		if (type != -1) {
			hand.type = type;
			hand.location.x = box.x;
			hand.location.y = box.y;
			return hand;
		}

	}
	//imshow("sssssssssss", front);
	//waitKey(0);
	//imshow("sssssssss5555ss", only_object);
	//waitKey(0);
	return hand;
}