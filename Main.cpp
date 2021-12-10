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

string const video_name_path = "hand.mp4";
int const skip_frames = 3;/////////////////////////////////////////////
Scalar const box_color = Scalar(0, 0, 255);

int test = 0;

Mat ExtractBackground(VideoCapture& video);
void PrepareImage(Mat& image, bool is_background);
Mat BackgroundRemover(const Mat& front, const Mat& back);
vector<vector<Point>> FindImageContours(const Mat& object);
bool CompareContourAreas(const vector<Point> contour1, const vector<Point> contour2);
Hand SearchForHand(const Mat& front, const vector<vector<Point>>& contours, Rect& box);
void PrintHandType(Mat& frame, const int h_type);
void PrintHandLocation(Mat& frame, const Point hand_pos);
int HandMovementDirection(const Hand& current, const Hand& previous);
Mat MovementDirectionShape(const int direction);

//// Main Method - Video
//// Precondition: hand.mp4 exists in the code directory and is a valid mp4 video file.
//// Postcondition: an image for each hand frame gets shown indicating hand position, location, and change from previous frame
/*
int main(int argc, char* argv[]) {
	VideoCapture cap(video_name_path);
	if (!cap.isOpened()) return -1;

	int const frame_width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
	int const frame_height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
	Mat frame;
	Mat background = ExtractBackground(cap);
	//Mat background = imread("background.jpg");
	PrepareImage(background, true);
	Hand current_hand;
	Hand previous_hand;
	Mat original_frame(frame_height, frame_width, CV_8UC3);
	Mat front(frame_height, frame_width, CV_8UC3);


	imwrite("out_back.jpg", background);////////@@@@
	VideoWriter output_vid("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(frame_width, frame_height));

	int frame_num = 1;
	int previous_shape_type = -1;
	Rect prev_box;

	while (true) {
		cap >> frame;				// Reads in image frame
		if (!frame.data) break;	// if there's no more frames then break
		if (frame_num % skip_frames == 0) {	//decreases the number of frames being read in
			original_frame = frame.clone();


			PrepareImage(frame, true);
			front = BackgroundRemover(frame, background);
			vector<vector<Point>> contours = FindImageContours(front);
			sort(contours.begin(), contours.end(), CompareContourAreas);
			Rect box;
			current_hand = SearchForHand(front, contours, box);



			//Print info to screen
			PrintHandType(original_frame, current_hand.type);
			PrintHandLocation(original_frame, current_hand.location);
			int shape_type = HandMovementDirection(current_hand, previous_hand);
			Mat shape = MovementDirectionShape(shape_type);
			shape.copyTo(original_frame(Rect(0, 0, shape.cols, shape.rows)));
			if (current_hand.type != -1) {
				rectangle(original_frame, box, box_color, 2);
				prev_box = box;
			}


			previous_shape_type = shape_type;
			previous_hand = current_hand;


			//imshow("Video", original_frame);
			//waitKey(0);
			output_vid.write(original_frame);
			frame_num++;
		}
		else {
			PrintHandType(frame, previous_hand.type);
			PrintHandLocation(frame, previous_hand.location);
			Mat shape = MovementDirectionShape(previous_shape_type);
			shape.copyTo(frame(Rect(0, 0, shape.cols, shape.rows)));
			if (previous_hand.type != -1) {
				rectangle(frame, prev_box, box_color, 2);
			}
			output_vid.write(frame);
			frame_num++;
			//imshow("56", frame);
			//waitKey(0);
		}
	}
	output_vid.release();
	cap.release();
	return 0;
}
*/

//// Main Method - Picture
//// Precondition: front.jpg and background.jpg exist in the code directory and are valid JPEG files.
//// Postcondition: an image gets shown indicating hand position and location.
int main() {
	Mat frame = imread("front.jpg");
	Mat background = imread("background.jpg");
	PrepareImage(background, true);
	Hand current_hand;
	Hand previous_hand;

	Mat original_frame(background.rows, background.cols, CV_8UC3);
	Mat front(background.rows, background.cols, CV_8UC3);


	original_frame = frame.clone();
	PrepareImage(frame, false);
	front = BackgroundRemover(frame, background);

	imwrite("qweedsdf.jpg", front);

	vector<vector<Point>> contours = FindImageContours(front);
	sort(contours.begin(), contours.end(), CompareContourAreas);
	Rect box;

	current_hand = SearchForHand(front, contours, box);

	//Hand is either detected or not	//Print info to screen
	PrintHandType(original_frame, current_hand.type);
	PrintHandLocation(original_frame, current_hand.location);
	Mat shape = MovementDirectionShape(HandMovementDirection(current_hand, previous_hand));
	shape.copyTo(original_frame(Rect(0, 0, shape.cols, shape.rows)));
	if (current_hand.type != -1) {
		rectangle(original_frame, box, Scalar(0, 255, 0), 2);
	}

	imshow("ddsadcewfwefw", original_frame);
	waitKey(0);
	imwrite("done.jpg", original_frame);
}
