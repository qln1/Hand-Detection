// 
// 
// 
// Author: Quintin Nguyen, 

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <cmath>
#include <opencv2/core/types.hpp>
#include <vector>
using namespace cv;
using namespace std;

Scalar const text_color = { 255, 0 ,0 };
string const path = "hand.mp4";
double const contrast_num = 1.5;
int const gaus_blur = 21;
int const background_remover_thresh = 30;
int const median_blur = 15;
int const num_template_files = 8;
int const similarity_threshold = 75;

struct Hand {
	Point location;
	int type;
};











// FixComputedRow
// Precondition: Parameter is passed in correctly
// Postcondition: Will return num as an int and makes sure that
//                num will not go out of bounds.
int FixComputedRow(Mat search, int row) {
	if (row >= search.rows) {
		row = search.rows - 1;
	}
	else if (row < 0) {
		row = 0;
	}
	return row;
}

// FixComputedCol
// Precondition: Parameter is passed in correctly
// Postcondition: Will return num as an int and makes sure that
//                num will not go out of bounds.
int FixComputedCol(Mat search, int col) {
	if (col >= search.cols) {
		col = search.cols - 1;
	}
	else if (col < 0) {
		col = 0;
	}
	return col;
}

// IsPointAnEdge
// Precondition: Parameter is passed in correctly. search is a edge image
// Postcondition: Will tell if the point (directly overlapping or off by
//                one pixel, including diagonally) is or is not an edge 
bool IsPointAnEdge(Mat search, int current_row, int current_col) {
	int row_pos = FixComputedRow(search, current_row + 1);
	int row_neg = FixComputedRow(search, current_row - 1);

	int col_pos = FixComputedCol(search, current_col + 1);
	int col_neg = FixComputedCol(search, current_col - 1);

	if (search.at<uchar>(current_row, current_col) != 0 ||
		search.at<uchar>(current_row, col_pos) != 0 ||
		search.at<uchar>(current_row, col_neg) != 0 ||
		search.at<uchar>(row_pos, current_col) != 0 ||
		search.at<uchar>(row_neg, current_col) != 0 ||
		search.at<uchar>(row_pos, col_pos) != 0 ||
		search.at<uchar>(row_pos, col_neg) != 0 ||
		search.at<uchar>(row_neg, col_pos) != 0 ||
		search.at<uchar>(row_neg, col_neg) != 0) {
		return true;
	}
	return false;
}

//// HowSimilarImagesAre
//// Precondition: 
//// Postcondition: Returns a 
//float HowSimilarImagesAre(Mat search, Mat temp) {
//	float end_sum = 0;
//	float white_spot = 0;
//	for (int t_row = 0; t_row < temp.rows; t_row++) {
//		for (int t_col = 0; t_col < temp.cols; t_col++) {
//			if (temp.at<uchar>(t_row, t_col) != 0) {
//				white_spot++;
//				if (IsPointAnEdge(search, t_row, t_col)) {
//					end_sum++;
//				}
//			}
//		}
//	}
//	return ((end_sum / white_spot) * 100);
//}


float HowSimilarImagesAre(Mat search, Mat temp) {
	//Ptr<Feature2D> f2d = SIFT::create();
	//vector<KeyPoint> keypoints_template;
	//vector<KeyPoint> keypoints_search;
	//f2d->detect(temp, keypoints_template);
	//f2d->detect(search, keypoints_search);

	//Mat descriptors_1, descriptors_2;
	//f2d->compute(temp, keypoints_template, descriptors_1);
	//f2d->compute(search, keypoints_search, descriptors_2);

	//BFMatcher matcher;
	//std::vector< DMatch > matches;
	//matcher.match(descriptors_1, descriptors_2, matches);

	//Mat output;
	////cv::drawKeypoints(temp, keypoints_template, output);
	//matchTemplate(search, temp, output, TM_CCOEFF);
	//imshow("aaaaaaaaaa", output);
	return 1.0;
}















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
Mat ModifyContrast(Mat pic) {
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

	return pic;
}





// SearchForTemplate
// Precondition: search.jpg and temp.jpg exists in the code directory
//               and is a valid JPG.
// Postcondition: Returns a vector of the row and col for the search image
//                that contains the most pixels that matches the template
Mat BackgroundRemover(Mat front, Mat back) {
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
				abs(front_color_r - back_color_r) < background_remover_thresh ) {	//Very similar
				output.at<uchar>(row, col) = 0;
			}
			else {	//Not similar at all	Hand detected
				output.at<uchar>(row, col) = 255;
			}
		}
	}
	return output;
}












//int getMaxAreaContourId(vector <vector<cv::Point>> contours) {
//    double maxArea = 0;
//    int maxAreaContourId = -1;
//    for (int j = 0; j < contours.size(); j++) {
//        double newArea = cv::contourArea(contours.at(j));
//        if (newArea > maxArea) {
//            maxArea = newArea;
//            maxAreaContourId = j;
//			bounding_rect = boundingRect(contours[j]);
//        } // End if
//    } // End for
//    return maxAreaContourId;
//} // End function


int FindBiggestContour(vector <vector<Point>> contours, Rect& bounding_rect) {
	double biggest_area = -1;
	int biggest_contour = -1;
	int i = 0;
	while(i < contours.size()) {
		double current_area = contourArea(contours.at(i));
		if (current_area >= biggest_area) {
			biggest_area = current_area;
			biggest_contour = i;
			
		}
		i++;
	}
	bounding_rect = boundingRect(contours[biggest_contour]);
	return biggest_contour;
}








Mat MovementDirectionShape(int direction, bool hand_detected) {
	Mat shape;
	if (hand_detected) {
		if (direction == 0) {
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
			else { }	//Right
		}
	}
	else {
		shape = imread("none.jpg");
	}
	return shape;
}




//void CreateRectangle(Mat frame) {
//	int x = 0;
//	int y = 0;
//	int width = 10;
//	int height = 20;
//
//
//	// our rectangle...
//	cv::Rect rect(x, y, width, height);
//	// and its top left corner...
//	cv::Point pt1(x, y);
//	// and its bottom right corner.
//	cv::Point pt2(x + width, y + height);
//
//
//	cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0));
//}










//Given a coordinate, it will put the text on the frame
//No hands means 
		//hand_pos = -1, -1
		//h_type == 8
void PrintHandLocation(Mat frame, bool hand_detected, Point hand_pos, int h_type) {
	string hand_location = "Hand Location: (" + to_string(hand_pos.x) + ", " + to_string(hand_pos.y) + ")";
	putText(frame, hand_location, Point{ 3, frame.rows - 6 }, 1, 1.5, text_color, 2);
}



//Given a coordinate, it will put the text on the frame
//No hands means 
		//hand_pos = -1, -1
		//h_type == 8
void PrintHandType(Mat frame, bool hand_detected, Point hand_pos, int h_type) {
	string type;

	if (h_type == -1) {
		type = "No Hand Detected";
	}
	else if (h_type == 0) {
		type = "Ok Sign";
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
	else if (h_type == 6) {
		type = "Thumbs Up";
	}
	else if (h_type == 7) {
		type = "Thumbs Down";
	}

	string hand_type = "Hand Type: " + type;
	putText(frame, hand_type, Point{ 3, frame.rows - 12 }, 1, 1.5, text_color, 2);
}




void PrepareImages(Mat& image) {
	GaussianBlur(image, image, Size(gaus_blur, gaus_blur), 0);
	medianBlur(image, image, median_blur);
	image = ModifyContrast(image);
}



Mat ObtainFrontObject(Mat& object, Rect& bounding_rect) {
	Mat thresh;
	threshold(object, thresh, 90, 255, THRESH_BINARY);
	//imshow("s", object);
	//waitKey(0);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	Mat object_copy = object.clone();

	
	drawContours(object, contours, FindBiggestContour(contours, bounding_rect), Scalar(0, 255, 0), 2);

	rectangle(object_copy, bounding_rect, Scalar(0, 255, 0), 2);
	imshow("none 522222222222222222", object);
	waitKey(0);

	return object(bounding_rect);
}



Mat ObjectCropper(Mat& object) {
	float height = object.size().height;
	float width = object.size().width;
	if (height >= width) {
		if (height / width < 1.8) {}
		else {
			object = object(Range(0, height - (height / 3)), Range(0, width));
		}
	}
	else {
		if (width / height < 1.8) {}
		else {
			object = object(Range(0, width), Range(0, height));
		}
	}
	return object;
}




int TemplateMatchingWithObject(Mat object) {
	//Possible detect left vs right here first???

	int most_similar_file = -1;
	int highest_similar_value = -1;

	for (int i = 0; i < num_template_files; i++) {
		string file_name = "Templates\\" + to_string(i) + ".jpg";
		Mat templ = imread(file_name);
		int current_simi = HowSimilarImagesAre(object, templ);
		cout << i << " " << current_simi << endl;
		if (current_simi > highest_similar_value) {
			highest_similar_value = current_simi;
			most_similar_file = i;
		}
	}
	if (round(highest_similar_value) >= similarity_threshold) {	//matches a template
		return most_similar_file;
	}
	else {	//fails to detect hand
		return -1;
	}
}





//// Main Method
//// Precondition: 
//// Postcondition: 
//int main(int argc, char* argv[]) {
//	//First determine what type of arrow/shape.   Display arrow/shape
//	//Mat arrow = CreateArrow(0, true);
//	//arrow.copyTo(background(Rect(10, 10, arrow.cols, arrow.rows)));
//
//
//
//
//
//
//
// 
//	VideoCapture cap(path);
//	if (!cap.isOpened()) return -1;
//
//	Mat frame;
//	Mat background;
//	Point previous_hand_loc;
//	bool first_frame = true;
//
//	int const frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
//	int const frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
//	VideoWriter output_vid("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(frame_width, frame_height));
//
//
//
//
//
//	while (true) {
//		cap >> frame;				// Reads in image frame
//		if (!frame.data) break;	// if there's no more frames then break
//
//
//
//
//
//
//		//Do STUFF WITH FRAME
//		if (first_frame) {
//			cvtColor(frame, frame, COLOR_BGR2GRAY, 0);
//			GaussianBlur(frame, frame, Size(7, 7), 2.0, 2.0);
//			Canny(frame, frame, 20, 60);
//			background = frame;
//			first_frame = false;
//		}
//		else {
//			//
//
//			//AnalyzeSegment
//			//
//		}
//
//
//
//		imshow("Video", frame);
//		waitKey(30);
//		output_vid.write(frame);
//	}
//
//	output_vid.release();
//	cap.release();
//	destroyAllWindows();
//	return 0;
//}





int main() {
	Mat back = imread("background.jpg");
	Mat front = imread("front.jpg");

	PrepareImages(front);
	PrepareImages(back);
	
	//imshow("ascasc", front);
	//waitKey(0);






	
	//Mat hsv_front;
	//cvtColor(front, hsv_front, COLOR_BGR2HSV);
	//vector<Mat> channels_front;
	//split(front, channels_front);
	//Mat V_front = channels_front[2];

	//Mat hsv_back;
	//cvtColor(back, hsv_back, COLOR_BGR2HSV);
	//vector<Mat> channels_back;
	//split(back, channels_back);
	//Mat V_back = channels_back[2];








	//cvtColor(front, front, COLOR_BGR2GRAY);
	//Canny(V_front, V_front, 20, 60);
	//Canny(V_back, V_back, 20, 60);

	//imshow("ascasc", V_front);
	//waitKey(0);

	//imshow("ascasc", V_back);
	//waitKey(0);






	Mat object = BackgroundRemover(front, back);
	imshow("2222", object);
	waitKey(0);
	
	
	Rect bounding_rect;
	Mat only_object = ObtainFrontObject(object, bounding_rect);
	imshow("None approximation", only_object);
	waitKey(0);
	only_object = ObjectCropper(only_object);
	resize(only_object, only_object, Size(300, 490), INTER_LINEAR);
	imwrite("t.jpg", only_object);
	imshow("None approximation", only_object);
	waitKey(0);



	//only_object = imread("2.jpg");


	int type = TemplateMatchingWithObject(only_object);
	cout << type << endl;
	Hand hand;
	if (type != -1) {
		hand.type = type;
		hand.location.x = bounding_rect.x;
		hand.location.y = bounding_rect.y;

	}
	else {
		//Set previous hand to nothing
	}

}