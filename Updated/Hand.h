// Contains the Hand struct for Hand Detection. Struct contains location of hand and position.
// Author: Quintin Nguyen, Akhil Lal, Matthew Cho

#pragma once
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

struct Hand {
	Point location = Point(-1, -1);
	int type = -1;
};
