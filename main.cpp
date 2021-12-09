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

int test = 0;

struct Hand {
	Point location = Point(-1, -1);
	int type = -1;
};











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













// PrepareImage
// Precondition: Parameters and image is properly formatted, passed in correctly and colored
// Postcondition: Will modify image by putting various blurrs and filters on top. image will
//                be modified slightly differently depending if it is a background or not.
void PrepareImage(Mat& image, bool is_background) {
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

// MovementDirectionShape
// Precondition: Parameter is properly formatted and passed in correctly
// Postcondition: Will return an image based on the direction that was passed in
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


// PrintHandLocation
// Precondition: Parameters are properly formatted and passed in correctly
// Postcondition: Will write the hand location on the passed in frame
void PrintHandLocation(Mat& frame, const Point hand_pos) {
	string hand_location = "Hand Location: (" + to_string(hand_pos.x) + ", " + to_string(hand_pos.y) + ")";
	putText(frame, hand_location, Point{ 3, frame.rows - 6 }, 1, 1.5, text_color, 2);
}

//Given a coordinate, it will put the text on the frame
//No hands means
		//hand_pos = -1, -1
		//h_type == 8
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














vector<vector<Point>> FindImageContours(const Mat& object) {
	Mat thresh;
	threshold(object, thresh, 90, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	return contours;
}

int FindNthBiggestContour(const vector<vector<Point>>& contours, Rect& box, const int n) {
	int index = contours.size() - n;
	if (contourArea(contours[index]) >= min_contour_area) {
		box = boundingRect(contours[index]);
		return index;
	}
	return -1;
}

bool CompareContourAreas(const vector<Point> contour1, const vector<Point> contour2) {///////////////////////////////////////////////
	double i = fabs(contourArea(Mat(contour1)));
	double j = fabs(contourArea(Mat(contour2)));
	return (i < j);
}






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



	//for () {

	//}


	//if (true_minima.size() - true_maxima.size() == 1 &&
	//if (true_minima.size() > true_maxima.size() &&
	if(true_minima.size() > 0 && true_maxima.size() < 6) {
		if (true_minima.size() == true_maxima.size()) {
			return (true_maxima.size() + 1);
		}
		else if (true_minima.size() - true_maxima.size() == 1) {
			return true_minima.size();
		}
	}
	return -1;
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












//// Main Method - Video
//// Precondition: hand.mp4 exists in the code directory and is a valid mp4 video file. 
//// Postcondition: an image for each hand frame gets shown indicating hand position, location, and change from previous frame
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















/////////////////////////////////////////////////////PICTURE ONLY
//// Main Method - Picture
//// Precondition: front.jpg and background.jpg exist in the code directory and are valid JPEG files. 
//// Postcondition: an image gets shown indicating hand position and location.
//int main() {
//	Mat frame = imread("front.jpg");
//	Mat background = imread("background.jpg");
//	PrepareImage(background, true);
//	Hand current_hand;
//	Hand previous_hand;
//
//
//	Mat original_frame(background.rows, background.cols, CV_8UC3);
//	Mat front(background.rows, background.cols, CV_8UC3);
//
//
//	original_frame = frame.clone();
//	PrepareImage(frame, false);
//	front = BackgroundRemover(frame, background);
//
//	imwrite("qweedsdf.jpg", front);
//
//	vector<vector<Point>> contours = FindImageContours(front);
//	sort(contours.begin(), contours.end(), CompareContourAreas);
//	Rect box;
//
//	current_hand = SearchForHand(front, contours, box);
//
//	//Hand is either detected or not	//Print info to screen
//	PrintHandType(original_frame, current_hand.type);
//	PrintHandLocation(original_frame, current_hand.location);
//	Mat shape = MovementDirectionShape(HandMovementDirection(current_hand, previous_hand));
//	shape.copyTo(original_frame(Rect(0, 0, shape.cols, shape.rows)));
//	if (current_hand.type != -1) {
//		rectangle(original_frame, box, Scalar(0, 255, 0), 2);
//	}
//
//	imshow("ddsadcewfwefw", original_frame);
//	waitKey(0);
//	imwrite("done.jpg", original_frame);
//}
