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
	std::vector< std::vector<cv::Point2f> > imagePoints; // Vector of vectors containing detected corners of all images.
	std::vector<String> filenames; //Vector of loaded images.
	std::vector<cv::Vec3f> objPoints; //Vector of x, y, z coordinates of inner corners of an image.	
	std::vector< std::vector<cv::Vec3f> > objectPoints; // Vector of objPoints of all images.
	std::vector<cv::Mat> rvecs; //Output vector of rotation vectors estimated for each pattern view.
	std::vector<cv::Mat> tvecs; //Output vector of translation vectors estimated for each pattern view.

	//Termination criteria for the iterative optimization algorithm.
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::Type::EPS | cv::TermCriteria::Type::MAX_ITER, 30, 0.00001);
	cv::TermCriteria criteria_calib = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.00001);

	// Prepare objPoints: each image has the same checkerboard pattern structure.
	for (int c = 1; c <= 12; c++)
	{
		for (int r = 1; r <= 8; r++)
		{
			objPoints.push_back(cv::Vec3f(c * SQ_SIZE, r * SQ_SIZE, 0));
		}
	}
	
	//Load images from folder
	String folder = "C:\\Users\\david\\source\\repos\\Camera calibration\\Huawei-Pietro\\Huawei-Pietro"; 
	glob(folder, filenames); 
	
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
	cv::resize(img, img_resized, cv::Size(), 0.5, 0.5);// just for better visualization
	cv::namedWindow(folder);
	cv::imshow(folder, img_resized);
	cv::waitKey(20);
	cv::destroyWindow(folder);	

	//fill objectPoints container.
	objectPoints.push_back(objPoints);
	imageSize = img.size();
	std::cout << imageSize << std::endl;
	std::cout << "DONE\n" << std::endl;

	}	
	
	distCoeffs = Mat::zeros(8, 1, CV_64F);
	cameraMatrix = Mat::eye(3, 3, CV_64F);

	//Camera calibration
	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0, criteria_calib);

	/*Computes the reprojection error
	double computeReprojectionErrors(const vector<vector<Point3f> > & objectPoints, const vector<vector<Point2f> > & imagePoints,
		const vector<Mat> & rvecs, const vector<Mat> & tvecs, const Mat & cameraMatrix, const Mat & distCoeffs,
		vector<float> & perViewErrors);
	
		std::vector< std::vector<cv::Point2f> > imagePoints2;
		int i, totalPoints = 0;
		double totalErr = 0, err;
		perViewErrors.resize(objectPoints.size());

		for (i = 0; i < (int)objectPoints.size(); ++i)
		{
			projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2); //projects
			err = norm(Mat(imagePoints[i]), Mat(imagePoints2), noArray()); // difference

			int n = (int)objectPoints[i].size();
			perViewErrors[i] = (float)std::sqrt(err * err / n); // save for this view
			totalErr += err * err; // sum it up
			totalPoints += n;
		}

		return std::sqrt(totalErr / totalPoints); // calculate the arithmetical mean
		*/
	}


	



