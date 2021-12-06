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
using namespace cv;
using namespace std;

#define STAYING_STILL 0;
#define MOVE_DOWN 1;
#define MOVE_UP 2;
#define MOVE_LEFT 3;
#define MOVE_RIGHT 4;

Scalar const text_color = { 0, 255 ,0 };
string const video_name_path = "hand.mp4";
double const contrast_num = 1.25;
int const gaus_blur = 9;
int const background_remover_thresh = 18;
int const median_blur = 5;
string const template_path = "Templates\\hand";
int const skip_frames = 20;
int const min_contour_area = 14000;
int const similarity_threshold = 11;
int const movement_threshold = 10;
int const min_hessian = 400;
float const ratio_thresh = 0.5;
int const number_random_frames = 30;

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

void PrepareImages(Mat& image) {
	GaussianBlur(image, image, Size(gaus_blur, gaus_blur), 3);
	medianBlur(image, image, median_blur);
	ModifyContrast(image);

	vector<Mat> channels_front;
	split(image, channels_front);
	image = channels_front[0];
	imshow("asdfsadfsdfsdfsdfsdasdfafwfwfwewef", image);
	waitKey(0);
}
















int HowSimilarImagesAre(const Mat& hand, const Mat& front) {
	Ptr<SIFT> detector = SIFT::create(min_hessian);
	vector<KeyPoint> keypoints_template, keypoints_search;
	Mat descriptor_template, descriptor_search;
	detector->detectAndCompute(hand, noArray(), keypoints_template, descriptor_template);
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











Mat BackgroundRemover(const Mat& front, const Mat& back) {
	Mat output(back.rows, back.cols, CV_8U);
	for (int row = 0; row < back.rows; row++) {
		for (int col = 0; col < back.cols; col++) {
			int front_color = front.at<uchar>(row, col);
			int back_color = back.at<uchar>(row, col);
			if (abs(front_color - back_color) < background_remover_thresh) {	//Very similar
				output.at<uchar>(row, col) = 0;
			}
			else {	//Not similar. Object here
				output.at<uchar>(row, col) = 255;
			}
		}
	}
	return output;
}

















int HandMovementDirection(const Hand& current, const Hand& previous) {
	int change_in_x = current.location.x - previous.location.x;
	int change_in_y = current.location.y - previous.location.y;
	if (current.type == -1 || previous.type == -1) {
		return -1;
	}
	if (change_in_x >= change_in_y) {
		if (change_in_x > movement_threshold) {
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
		if (change_in_y > movement_threshold) {
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


//Given a coordinate, it will put the text on the frame
//No hands means
		//hand_pos = -1, -1
		//h_type == 8
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
	putText(frame, hand_type, Point{ 3, frame.rows - 30}, 1, 1.5, text_color, 2);
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











Hand SearchForHand(const Mat& frame, const Mat& front, const vector<vector<Point>>& contours, Rect& box) {
	Hand hand;
	Mat only_object;
	for (int i = 1; i <= contours.size(); i++) {
		int contour_index = FindNthBiggestContour(contours, box, i);
		if (contour_index == -1) {
			break;
		}
		drawContours(front, contours, contour_index, Scalar(0, 255, 0), 2);
		rectangle(front, box, Scalar(0, 255, 0), 2); //draws rectangle/////////////////////////////////////////

		only_object = frame(box);
		imshow("asdcxdf", only_object);

		int type = TemplateMatchingWithObject(only_object);
		cout << "TYPEE!! " << type << endl;
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












//// Main Method
//// Precondition:
//// Postcondition:
//int main(int argc, char* argv[]) {
//	VideoCapture cap(video_name_path);
//	if (!cap.isOpened()) return -1;
//
//	int const frame_width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
//	int const frame_height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
//	Mat frame;
//	Mat background = ExtractBackground(cap);
//	//Mat background = imread("out_back.jpg");
//	PrepareImages(background);
//	Hand current_hand;
//	Hand previous_hand;
//	Mat original_frame(frame_height, frame_width, CV_8UC3);
//	Mat front(frame_height, frame_width, CV_8UC3);
//
//	//imshow("Video", background);
//	//imwrite("out_back.jpg", background);
//	//waitKey(0);
//
//	////////bool first_frame = true;
//
//
//	VideoWriter output_vid("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(frame_width, frame_height));
//	int frame_num = 1;	//1106 frames per 40sec video
//	while (true) {
//		if (frame_num % skip_frames == 0) {	//decreases the number of frames being read in
//			cap >> frame;				// Reads in image frame
//			if (!frame.data) break;	// if there's no more frames then break
//
//			//cout << "sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss" << endl;
//			//imshow("Video", frame);
//			//imwrite("out_back.jpg", background);
//			//waitKey(0);
//			original_frame = frame.clone();
//			PrepareImages(frame);
//			front = BackgroundRemover(frame, background);
//
//
//
//			vector<vector<Point>> contours = FindImageContours(front);
//			sort(contours.begin(), contours.end(), CompareContourAreas);
//			Rect box;
//
//
//
//			current_hand = SearchForHand(front, contours, box);
//
//
//
//			//Hand is either detected or not	//Print info to screen
//			PrintHandType(original_frame, current_hand.type);
//			PrintHandLocation(original_frame, current_hand.location);
//			Mat shape = MovementDirectionShape(HandMovementDirection(current_hand, previous_hand));
//			shape.copyTo(original_frame(Rect(100, 200, shape.cols, shape.rows)));
//			if (current_hand.type != -1) {
//				rectangle(original_frame, box, Scalar(0, 255, 0), 2);
//			}
//
//
//			previous_hand = current_hand;
//
//
//			//imshow("Video", frame);
//			//waitKey(30);
//			output_vid.write(original_frame);
//			frame_num++;
//		}
//		else {
//			frame_num++;
//		}
//	}
//
//	output_vid.release();
//	cap.release();
//	//destroyAllWindows();
//	return 0;
//}





/////////////////////////////////////////////////////PICTURE ONLY

int main() {
	Mat frame = imread("front.jpg");
	Mat background = imread("background.jpg");
	PrepareImages(background);
	Hand current_hand;
	Hand previous_hand;////


	Mat original_frame(background.rows, background.cols, CV_8UC3);
	Mat front(background.rows, background.cols, CV_8UC3);


	original_frame = frame.clone();
	PrepareImages(frame);
	front = BackgroundRemover(frame, background);
	imshow("asdax", front);
	waitKey(0);
	vector<vector<Point>> contours = FindImageContours(front);
	sort(contours.begin(), contours.end(), CompareContourAreas);
	Rect box;

	current_hand = SearchForHand(frame, front, contours, box);

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













//if (first_frame) {
//	cvtColor(frame, frame, COLOR_BGR2GRAY, 0);
//	GaussianBlur(frame, frame, Size(7, 7), 2.0, 2.0);
//	Canny(frame, frame, 20, 60);
//	background = frame;
//	first_frame = false;
//}
//else {
//	//


//int main() {
//	Mat back = imread("background.jpg");
//	Mat front = imread("front.jpg");
//
//	PrepareImages(front);
//	PrepareImages(back);
//
//	//imshow("ascasc", front);
//	//waitKey(0);
//
//  //Mat hsv_front;
//	//cvtColor(front, hsv_front, COLOR_BGR2HSV);
//	//vector<Mat> channels_front;
//	//split(front, channels_front);
//	//Mat V_front = channels_front[2];
//
//	//Mat hsv_back;
//	//cvtColor(back, hsv_back, COLOR_BGR2HSV);
//	//vector<Mat> channels_back;
//	//split(back, channels_back);
//	//Mat V_back = channels_back[2];
//
//  //cvtColor(front, front, COLOR_BGR2GRAY);
//	//Canny(V_front, V_front, 20, 60);
//	//Canny(V_back, V_back, 20, 60);
//
//	//imshow("ascasc", V_front);
//	//waitKey(0);
//
//	//imshow("ascasc", V_back);
//	//waitKey(0);
//
//	Mat object = BackgroundRemover(front, back);
//	imshow("2222", object);
//	waitKey(0);
//
//	Rect box;
//	Mat only_object = ObtainFrontObject(object, box);
//	imshow("objjjj", object);
//	waitKey(0);
//	//only_object = ObjectCropper(only_object);
//	//resize(only_object, only_object, Size(300, 490), INTER_LINEAR);
//	imwrite("t.jpg", only_object);
//	imshow("None approximation", only_object);
//	waitKey(0);
//
//	//only_object = imread("2.jpg");
//
//	int type = TemplateMatchingWithObject(only_object);
//	cout << "TYPEE!! " << type << endl;
//	Hand hand;
//	if (type != -1) {
//		hand.type = type;
//		hand.location.x = box.x;
//		hand.location.y = box.y;
//
//	}
//}

// Main just to test ExtractBackground()
//int main() {
//	VideoCapture cap;
//	cap.open("india.mp4");
//	Mat background = ExtractBackground(cap);
//	imwrite("background.jpg", background);
//}


//Mat ObjectCropper(Mat& object) {
//	float height = object.size().height;
//	float width = object.size().width;
//	if (height >= width) {
//		if (height / width < 1.8) {}
//		else {
//			object = object(Range(0, height - (height / 3)), Range(0, width));
//		}
//	}
//	else {
//		if (width / height < 1.8) {}
//		else {
//			object = object(Range(0, width), Range(0, height));
//		}
//	}
//	return object;
//}