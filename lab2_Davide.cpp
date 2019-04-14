#include <vector>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/calib3d.hpp>

#define SQ_SIZE 0.102 //meters

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

	//initializations
	cv::Mat img_resized;
	cv::Mat img;
	cv::Mat gray_image;
	cv::Mat cameraMatrix;
	cv::Size imageSize;
	cv::Mat distCoeffs;

	std::vector<cv::Point2f> corners; //Vector to be filled by the detected corners of an image.	
	std::vector<std::vector<cv::Point2f>> imagePoints; // Vector of vectors containing detected corners of all images.
	std::vector<String> filenames; //Vector of loaded images.
	std::vector<cv::Vec3f> objPoints; //Vector of x, y, z coordinates of inner corners of an image.	
	std::vector<std::vector<cv::Vec3f>> objectPoints; // Vector of objPoints of all images.
	std::vector<cv::Mat> rvecs; //Output vector of rotation vectors estimated for each pattern view.
	std::vector<cv::Mat> tvecs; //Output vector of translation vectors estimated for each pattern view.
	std::vector<double> error;

	//Termination criteria for the iterative optimization algorithm.
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::Type::EPS | cv::TermCriteria::Type::MAX_ITER, 30, 0.00001);
	cv::TermCriteria criteria_calib = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.00001);

	// Prepare objPoints: each image has the same checkerboard pattern structure.
	for (int c = 1; c <= 8; c++)
	{
		for (int r = 1; r <= 12; r++)
		{
			objPoints.push_back(cv::Vec3f(float(c) * SQ_SIZE, float(r) * SQ_SIZE, 0));
		}
	}

	
	//Load images from folder
	std::cout << "Loading images from folder, please wait . . .\n";
	String folder = "C:\\Users\\david\\source\\repos\\Camera calibration\\D3300\\D3300"; 
	glob(folder, filenames); 
	std::cout << "DONE\n\n\n";
	
	//Detects the checkerboard intersection in each image
	for (size_t i = 0; i < filenames.size(); ++i){

		Size patternsize(12, 8); //interior number of corners
		img = cv::imread(filenames[i]); //source image
		imageSize = img.size();
		cv::cvtColor(img, gray_image, COLOR_RGB2GRAY);//gray image conversion

		//Pattern detection
		bool patternfound = findChessboardCorners(img, patternsize, corners, CALIB_CB_ADAPTIVE_THRESH 
			+ CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		//function to refine the corner detections
		if (patternfound){	
		cv::cornerSubPix(gray_image, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria); 
		}
	
		imagePoints.push_back(corners);

		//draw and show the corners
		cv::drawChessboardCorners(img, patternsize, Mat(corners), patternfound);
		cv::namedWindow(folder);
		cv::imshow(folder, img);
		cv::waitKey(0);
		cv::destroyWindow(folder);	

		//fill objectPoints container.
		objectPoints.push_back(objPoints);
		imageSize = img.size();
		
		distCoeffs = Mat::zeros(8, 1, CV_64F);
		cameraMatrix = Mat::eye(3, 3, CV_64F);

		//Camera calibration
		double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0, criteria_calib);
		std::cout << "rms for image " << i+1 << " is: " << rms << "\n\n" << std::endl;
		error.push_back(rms);
		
	}	

	//Finding the best and worst image according to rms error computed above
	double max = 0;
	double min = 15;
	int i_max = -1;
	int i_min = -1;
	for (int i = 0; i < error.size(); i++) {
		if (error[i] < min) {
			min = error[i];
			i_min = i;
		} 
		if (error[i] > max) {
			max = error[i];
			i_max = i;
		} 
	}
	
	std::cout << "min error is: " << min << " found on image " << i_min+1 << std::endl;
	std::cout << "max error is: " << max << " found on image " << i_max+1 << std::endl;

	std::cout << "\n\n\n" << std::endl;
	return 0;
}
