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

double const contrast_num = 1.25;
int const sat_val = 15;
int const gaus_blur_size = 11;
int const gaus_blur_amount = 3;
int const median_blur = 7;
int const brightness_level = 12;
int const number_random_frames = 80;
int const background_remover_thresh = 18;

int FixComputedColor(double num);

// ModifyContrast
// Precondition: test.jpg exists in the code directory and is a valid JPG.
// Postcondition: output.jpg will be saved to disk with all of the contrast
//                being enhanced
void ModifyContrast(Mat& pic) {
	double ave_blue = 0;
	double ave_green = 0;
	double ave_red = 0;

	for (int row = 0; row < pic.rows; row++) {
		for (int col = 0; col < pic.cols; col++) {
			ave_blue += pic.at<Vec3b>(row, col)[0];
			ave_green += pic.at<Vec3b>(row, col)[1];
			ave_red += pic.at<Vec3b>(row, col)[2];
		}
	}

	ave_blue = ave_blue / pic.total();
	ave_green = ave_green / pic.total();
	ave_red = ave_red / pic.total();

	for (int row = 0; row < pic.rows; row++) {
		for (int col = 0; col < pic.cols; col++) {
			double new_blue = ave_blue - double(pic.at<Vec3b>(row, col)[0]);
			new_blue = ave_blue - (new_blue * contrast_num);

			double new_green = ave_green - double(pic.at<Vec3b>(row, col)[1]);
			new_green = ave_green - (new_green * contrast_num);

			double new_red = ave_red - double(pic.at<Vec3b>(row, col)[2]);
			new_red = ave_red - (new_red * contrast_num);

			pic.at<Vec3b>(row, col)[0] = FixComputedColor(new_blue);
			pic.at<Vec3b>(row, col)[1] = FixComputedColor(new_green);
			pic.at<Vec3b>(row, col)[2] = FixComputedColor(new_red);
		}
	}
}

// Increases the saturation in each pixel in the given image, using saturation value in HSV
// Preconditions: image is of the correct type and correctly allocated
// Postconditions: given image's saturation is increased by the global constant sat_val amount
void IncreaseSaturation(Mat& image) {
	Mat saturated;
	cvtColor(image, saturated, COLOR_BGR2HSV);
	for (int row = 0; row < saturated.rows; row++) {
		for (int col = 0; col < saturated.cols; col++) {
			double originalSaturation = (double)saturated.at<Vec3b>(row, col)[1];
			int newVal = FixComputedColor(originalSaturation + sat_val);
			saturated.at<Vec3b>(row, col)[1] = newVal;
		}
	}
	cvtColor(saturated, image, COLOR_HSV2BGR);
}

// PrepareImage
// Precondition: Parameters and image is properly formatted, passed in correctly and colored
// Postcondition: Will modify image by putting various blurrs and filters on top. image will
//                be modified slightly differently depending if it is a background or not.
void PrepareImage(Mat& image, bool is_background) {
	IncreaseSaturation(image);
	GaussianBlur(image, image, Size(gaus_blur_size, gaus_blur_size), gaus_blur_amount);
	medianBlur(image, image, median_blur);
	ModifyContrast(image);
	if (!is_background) {
		//saturation + 5;
		image.convertTo(image, -1, 1, brightness_level);
	}
	else {
		//saturation normal
	}
	//cvtColor(image, image, COLOR_BGR2HSV);
	//vector<Mat> channels_front;
	//split(image, channels_front);
	//image = channels_front[2];
	//imshow("sss", image);
	//waitKey(0);
	//imwrite(to_string(test) + ".jpg", image);
	//test++;
}

// Detects and extracts the background from given video
// preconditions: video is correctly formatted and allocated
// postconditions: the calculated background from the video is returned as a Mat
Mat ExtractBackground(VideoCapture& video) {
	const int frame_width = (int)video.get(CAP_PROP_FRAME_WIDTH);
	const int frame_height = (int)video.get(CAP_PROP_FRAME_HEIGHT);
	const int number_of_frames = (int)video.get(CAP_PROP_FRAME_COUNT);
	vector<int> random_frames;

	// Determine which random frames to use for background calculation
	for (int i = 0; i < number_random_frames; i++) {
		int random_frame = rand() % number_of_frames;
		while (find(random_frames.begin(), random_frames.end(), random_frame) != random_frames.end()) {
			random_frame = rand() % number_of_frames;
		}
		random_frames.push_back(random_frame);
	}

	Mat extracted_background(frame_height, frame_width, CV_8UC3, Scalar::all(0));
	Mat frame;
	int curFrame = 0;
	bool firstRandomFrame = true;
	vector<vector<vector<int>>> backgroundPixels(frame_height);

	// Go through each random frame and add its values to each pixel in extracted background
	for (;;) {
		video >> frame;
		if (frame.empty()) {
			break;
		}
		if (find(random_frames.begin(), random_frames.end(), curFrame) != random_frames.end()) {
			for (int row = 0; row < frame_height; row++) {
				if (firstRandomFrame) {
					vector<vector<int>> temp(frame_width);
					backgroundPixels.at(row) = temp;
				}
				for (int col = 0; col < frame_width; col++) {
					if (firstRandomFrame) {
						vector<int> colorValues(3);
						backgroundPixels.at(row).at(col) = colorValues;
						backgroundPixels.at(row).at(col).at(2) = frame.at<Vec3b>(row, col)[2];
						backgroundPixels.at(row).at(col).at(1) = frame.at<Vec3b>(row, col)[1];
						backgroundPixels.at(row).at(col).at(0) = frame.at<Vec3b>(row, col)[0];
					}
					else {
						backgroundPixels.at(row).at(col).at(2) += frame.at<Vec3b>(row, col)[2];
						backgroundPixels.at(row).at(col).at(1) += frame.at<Vec3b>(row, col)[1];
						backgroundPixels.at(row).at(col).at(0) += frame.at<Vec3b>(row, col)[0];
					}
				}
			}
			firstRandomFrame = false;
		}
		curFrame++;
	}

	// Average every pixel in background to get final background from video
	for (int row = 0; row < frame_height; row++) {
		for (int col = 0; col < frame_width; col++) {
			extracted_background.at<Vec3b>(row, col)[2] =
				FixComputedColor(backgroundPixels.at(row).at(col).at(2) / number_random_frames);
			extracted_background.at<Vec3b>(row, col)[1] =
				FixComputedColor(backgroundPixels.at(row).at(col).at(1) / number_random_frames);
			extracted_background.at<Vec3b>(row, col)[0] =
				FixComputedColor(backgroundPixels.at(row).at(col).at(0) / number_random_frames);
		}
	}
	video.set(CAP_PROP_POS_MSEC, 0);
	return extracted_background;
}

// BackgroundRemover
// Precondition: Parameters are properly formatted, passed in correctly and colored
// Postcondition: Will return a binary Matt where the white spots are the differences
//                between the 2 passed in Mats.
Mat BackgroundRemover(const Mat& front, const Mat& back) {
	Mat output(back.rows, back.cols, CV_8U);
	for (int row = 0; row < back.rows; row++) {
		for (int col = 0; col < back.cols; col++) {
			int front_color_b = front.at<Vec3b>(row, col)[0];
			int front_color_g = front.at<Vec3b>(row, col)[1];
			int front_color_r = front.at<Vec3b>(row, col)[2];
			int back_color_b = back.at<Vec3b>(row, col)[0];
			int back_color_g = back.at<Vec3b>(row, col)[1];
			int back_color_r = back.at<Vec3b>(row, col)[2];
			if (abs(front_color_b - back_color_b) < background_remover_thresh &&
				abs(front_color_g - back_color_g) < background_remover_thresh &&
				abs(front_color_r - back_color_r) < background_remover_thresh
				) {	//Very similar
				output.at<uchar>(row, col) = 0;
			}
			else {	//Not similar. Object here
				output.at<uchar>(row, col) = 255;
			}
		}
	}
	return output;
}

//Mat BackgroundRemover(const Mat& front, const Mat& back) {
//	Mat output(back.rows, back.cols, CV_8U);
//	for (int row = 0; row < back.rows; row++) {
//		for (int col = 0; col < back.cols; col++) {
//			int front_color = front.at<uchar>(row, col);
//			int back_color = back.at<uchar>(row, col);
//			if (abs(front_color - back_color) < background_remover_thresh) {	//Very similar
//				output.at<uchar>(row, col) = 0;
//			}
//			else {	//Not similar. Object here
//				output.at<uchar>(row, col) = 255;
//			}
//		}
//	}
//	return output;
//}
