#include "PanoramicUtils.h"
#include "PanoramicImage.h"

#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

#include "opencv2/opencv.hpp" 
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching.hpp>

using namespace std;
using namespace cv;

// Constructor
PanoramicImage::PanoramicImage(const vector<Mat> imageSet, const double focalLength, const double verticalFOV) {
	
	imSet = imageSet;
	focal_len = focalLength;
	vFOV = verticalFOV;

}
	


// 1.________________________________________________________________________________________________________

//Function that load images 
vector<Mat> PanoramicImage::loadImages(string pat, int& numImg) {

	vector<Mat> images; //Vector containing images
	Mat img;
	string path = pat;
	int totImages = 23;

	for (int i = 1; i <= totImages; i++) {
		path = "C:\\Users\\david\\source\\repos\\Keypoints_Descriptors_Matching\\dolomites\\" + to_string(i) + ".png";
		img = imread(path);
		images.push_back(img);
	}

	/*
	for (int i=0; i < totImages; i++) {
		namedWindow("current image");
		imshow("current image", images[i]);
		waitKey(0);
	}
	*/
	numImg = images.size();
	return images;
}

//Function that project the images on a cylinder surface 
void PanoramicImage::cyProj(PanoramicImage& panor) {

	for (int i = 0; i < panor.imSet.size(); i++) {
		PanoramicUtils::cylindricalProj(panor.imSet[i], panor.vFOV / 2);
	}
}

//___________________________________________________________________________________________________________



//2._________________________________________________________________________________________________________

//Function that extract the keypoints from the image 
void PanoramicImage::orbFeaturesExtractor(PanoramicImage panor, vector<vector<KeyPoint>>& totalKPoints, vector<Mat>& totalDescriptors) {

	vector<KeyPoint> keypoints;
	Mat descriptors;

	for (int i = 0; i < panor.imSet.size(); i++) {

	Ptr<FeatureDetector> detector = ORB::create();///Initiate ORB detector	
	Ptr<DescriptorExtractor> extractor = ORB::create();///Initiate ORB extractor

	detector->detect(panor.imSet[i], keypoints);
	extractor->compute(panor.imSet[i], keypoints, descriptors);
	totalKPoints.push_back(keypoints);
	totalDescriptors.push_back(descriptors);

	// cv::Mat output;
	// cv::drawKeypoints( panor.imSet[i], keypoints, output );
	// cv::namedWindow("ORB result");
	// cv::imshow( "ORB result", output );
	// cv::waitKey();

	}
	
}

//Function that ompute the match between the different features of each (consecutive) couple of images (a)
void PanoramicImage::matcher(PanoramicImage panor, vector<Mat>& totalDescriptors, vector<vector<DMatch>>& totalMatches) {

	vector<DMatch> matches;
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, false);
	
	for (int i = 0; i < panor.imSet.size() - 1; i++)
	{
		matcher->match(totalDescriptors[i], totalDescriptors[i + 1], matches);
		totalMatches.push_back(matches);
	}

	cout << "--->  MATCHES FOUND  <---" << endl;
}

//Function that refine the matches (b)
void PanoramicImage::matchesRefiner(vector<vector<DMatch>>& totalMatches, vector<vector<DMatch>>& totalRefinedMatches, double ratio) {
	
	float dist;
	int indexi = 0;
	int indexj = 0;
	float minDist; 
	
	vector<DMatch> refinedMatches;
	vector<float> minDistances; ///contains the minDist between the matches of (i) and (i+1) images.
	
	///Compute the minimum distance among matches for each couple of images_________________
	for (int i = 0; i < totalMatches.size(); i++)
	{
		minDist = totalMatches[i][0].distance;

		for (int j = 0; j < totalMatches[i].size(); j++)
		{
			dist = totalMatches[i][j].distance;

			if (dist < minDist)
			{
				minDist = dist;
				indexi = i;
				indexj = j;
			}
		}
		minDistances.push_back(minDist);
	}
	///_____________________________________________________________________________________

	///Refine using the ratio_______________________________________________________________
	if (ratio == 0){ratio = 3;}

	for (int i = 0; i < totalMatches.size(); i++){

		for (int j = 0; j < totalMatches[i].size(); j++){

			dist = totalMatches[i][j].distance;

			if (dist <= minDistances[i] * ratio){

				refinedMatches.push_back(totalMatches[i][j]);
			}
		}

		totalRefinedMatches.push_back(refinedMatches);
		refinedMatches.clear(); ///Throw it away

	}
	///_____________________________________________________________________________________

	cout << "--->  REFINING DONE  <---" << endl;

}

//Function that find the translation between the images (c)
void PanoramicImage::inliersRetriever(vector<vector<KeyPoint>>& totalKPoints, vector<vector<DMatch>>& totalRefinedMatches, vector<vector<DMatch>>& totalInliersGoodMatches) {

	std::vector<cv::DMatch> inliersGoodMatches; ///will fill the output vector totalInliersGoodMatches

	///Parameter for findHomography() function
	std::vector<cv::Point2f> points1, points2;
	cv::Mat hmask;
	std::vector<cv::Mat> totalhmask;

	///Fill totalHmask______________________________________________________________________
	for (int i = 0; i < totalKPoints.size() - 1; i++){

		for (int j = 0; j < totalRefinedMatches[i].size(); j++){

			points1.push_back(totalKPoints[i][totalRefinedMatches[i][j].queryIdx].pt);
			points2.push_back(totalKPoints[i + 1][totalRefinedMatches[i][j].trainIdx].pt);
		}

		findHomography(points1, points2, hmask, RANSAC);
		totalhmask.push_back(hmask);

		points1.clear(); ///Throw it away
		points2.clear(); ///Throw it away
	}
	///_____________________________________________________________________________________

	///Fill totalInliersGoodMatches_________________________________________________________
	for (int i = 0; i < totalRefinedMatches.size(); i++){

		for (int j = 0; j < totalRefinedMatches[i].size(); j++){

			if ((int)totalhmask[i].Mat::at<uchar>(j, 0)){

				inliersGoodMatches.push_back(totalRefinedMatches[i][j]);
			}
		}

		totalInliersGoodMatches.push_back(inliersGoodMatches);
		inliersGoodMatches.clear(); ///Throw it away

	}
	///_____________________________________________________________________________________

	cout << "--->  INLIERS RETRIVED  <---" << endl;
}

//___________________________________________________________________________________________________________



//3._________________________________________________________________________________________________________

void PanoramicImage::mergeImg(PanoramicImage panor, Mat& panoramic, vector<float>& totalMeanDist, vector<vector<float>>& totalDistance, vector<vector<DMatch>>& totalInliersGoodMatches) {

	///Compute the mean distance between a couples of images
	float dist;
	float numInliers;

	for (int i = 0; i < totalInliersGoodMatches.size(); i++){

		dist = 0;
		numInliers = totalInliersGoodMatches[i].size();

		for (int j = 0; j < numInliers; j++){

			dist = dist + totalDistance[i][j];

		}

		dist = dist / numInliers;
		totalMeanDist.push_back(dist);
	}

	///Apply translation of mean distance value
	Mat shiftMat(2, 3, CV_64F, Scalar(0.0));

	shiftMat.Mat::at<double>(0, 0) = 1;
	shiftMat.Mat::at<double>(1, 1) = 1;
	shiftMat.Mat::at<double>(0, 1) = 0;
	shiftMat.Mat::at<double>(1, 0) = 0;
	shiftMat.Mat::at<double>(1, 2) = 0;

	Mat dst;
	panoramic = panor.imSet[0];

	for (int i = 0; i < panor.imSet.size() - 1; i++){

		shiftMat.cv::Mat::at<double>(0, 2) = - totalMeanDist[i];
		warpAffine(panor.imSet[i + 1], dst, shiftMat, Size(panor.imSet[i + 1].cols - totalMeanDist[i], panor.imSet[i + 1].rows), INTER_CUBIC, BORDER_CONSTANT, Scalar());
		hconcat(panoramic, dst, panoramic);

	}

	cout << "SHOWING PANORAMIC IN 3... 2... 1..." << endl ;
	cv::namedWindow("panoramic");
	imshow("panoramic", panoramic);
	cv::waitKey(0);


}

//___________________________________________________________________________________________________________
