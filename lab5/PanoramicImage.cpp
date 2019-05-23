#include "PanoramicUtils.h"
#include "PanoramicImage.h"

#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <string>

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
void PanoramicImage::loadImages(string path, int& numImg, PanoramicImage& panor) {

	int totImages = numImg;
	Mat image;

	for (int i = 1; i <= totImages; i++) {

		string index = to_string(i);
		string here = path + index + ".png";

		//////////////////////////////////////////////////////
		//cout << "This image's path is: " << here << endl; //
		//////////////////////////////////////////////////////

		image = imread(here);
		panor.imSet.push_back(image);
		here.clear();
	}

}

//Function that project the images on a cylinder surface 
void PanoramicImage::cyProj(PanoramicImage panor) {

	for (int i = 0; i < panor.imSet.size(); i++) {
		panor.imSet[i] = PanoramicUtils::cylindricalProj(panor.imSet[i], panor.vFOV / 2);
	}
}

//___________________________________________________________________________________________________________





//2._________________________________________________________________________________________________________

//Function that extract the keypoints from the image 
void PanoramicImage::orbFeaturesExtractor(PanoramicImage panor, vector<vector<KeyPoint>> & totalKPoints, vector<Mat> & totalDescriptors) {

	vector<KeyPoint> keypoints;
	Mat descriptors;

	for (int i = 0; i < panor.imSet.size(); i++) {

		Ptr<FeatureDetector> detector = ORB::create(2000); ///Initiate ORB detector	

		//////////////////////////////////////////
		//imshow(to_string(i), panor.imSet[i]); //
		//waitKey(0);                           //
		//////////////////////////////////////////

		detector->detectAndCompute(panor.imSet[i], Mat(), keypoints, descriptors);

		///////////////////////////////////////////
		//cout << descriptors.size() << endl;    //
		//cout << keypoints.size() << endl;      //
		///////////////////////////////////////////

		totalKPoints.push_back(keypoints);
		totalDescriptors.push_back(descriptors);
		keypoints.clear(); ///Throw it away

	}

}

//Function that ompute the match between the different features of each (consecutive) couple of images (a)
void PanoramicImage::matcher(PanoramicImage panor, vector<Mat> & totalDescriptors, vector<vector<DMatch>> & totalMatches) {

	vector<DMatch> matches;
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, false);

	for (int i = 0; i < panor.imSet.size() - 1; i++)
	{
		matcher->match(totalDescriptors[i], totalDescriptors[i + 1], matches);
		totalMatches.push_back(matches);
	}

	matches.clear(); ///Throw it away

	cout << "POCESSING: \n\n" << "--->  MATCHES FOUND     <---" << endl;
}

//Function that refine the matches (b)
void PanoramicImage::matchesRefiner(vector<vector<DMatch>> & totalMatches, vector<vector<DMatch>> & totalRefinedMatches, double ratio) {

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
	if (ratio == 0) { ratio = 3; }

	for (int i = 0; i < totalMatches.size(); i++) {

		for (int j = 0; j < totalMatches[i].size(); j++) {

			dist = totalMatches[i][j].distance;

			if (dist <= minDistances[i] * ratio) {

				refinedMatches.push_back(totalMatches[i][j]);
			}
		}

		totalRefinedMatches.push_back(refinedMatches);
		refinedMatches.clear(); ///Throw it away

	}
	///_____________________________________________________________________________________

	cout << "--->  REFINING DONE     <---" << endl;

}

//Function that find the translation between the images (c)
void PanoramicImage::inliersRetriever(vector<vector<KeyPoint>> & totalKPoints, vector<vector<DMatch>> & totalRefinedMatches, vector<vector<DMatch>> & totalInliersGoodMatches) {

	vector<DMatch> inliersGoodMatches; ///will fill the output vector totalInliersGoodMatches

	///Parameter for findHomography() function
	vector<Point2f> points1, points2;
	Mat hmask;
	vector<Mat> totalhmask;

	///Fill totalHmask______________________________________________________________________
	for (int i = 0; i < totalKPoints.size() - 1; i++) {

		for (int j = 0; j < totalRefinedMatches[i].size(); j++) {

			points1.push_back(totalKPoints[i][totalRefinedMatches[i][j].queryIdx].pt);
			points2.push_back(totalKPoints[i + 1][totalRefinedMatches[i][j].trainIdx].pt);
		}

		findHomography(points1, points2, RANSAC, 3, hmask);
		totalhmask.push_back(hmask);

		points1.clear(); ///Throw it away
		points2.clear(); ///Throw it away
	}
	///_____________________________________________________________________________________

	///Fill totalInliersGoodMatches_________________________________________________________
	for (int i = 0; i < totalRefinedMatches.size(); i++) {

		for (int j = 0; j < totalRefinedMatches[i].size(); j++) {

			if ((int)totalhmask[i].Mat::at<uchar>(j, 0)) {

				inliersGoodMatches.push_back(totalRefinedMatches[i][j]);
			}
		}

		totalInliersGoodMatches.push_back(inliersGoodMatches);
		inliersGoodMatches.clear(); ///Throw it away

	}
	///_____________________________________________________________________________________

	cout << "--->  INLIERS RETRIVED  <---" << "\n" << endl;
}

//___________________________________________________________________________________________________________





//3._________________________________________________________________________________________________________

void PanoramicImage::mergeImg(PanoramicImage panor, Mat & panoramic, vector<float> & totalMeanDist, vector<vector<float>> & totalDistance, vector<vector<DMatch>> & totalInliersGoodMatches) {

	///Compute the mean distance between a couples of images
	float dist;
	float numInliers;

	for (int i = 0; i < totalInliersGoodMatches.size(); i++) {

		dist = 0;
		numInliers = totalInliersGoodMatches[i].size();

		for (int j = 0; j < numInliers; j++) {

			dist = dist + totalDistance[i][j];

		}

		dist = dist / numInliers;
		totalMeanDist.push_back(dist);
	}

	///Apply translation 
	Mat shiftMat(2, 3, CV_64F, Scalar(0.0));

	shiftMat.Mat::at<double>(0, 0) = 1;
	shiftMat.Mat::at<double>(1, 1) = 1;
	shiftMat.Mat::at<double>(0, 1) = 0;
	shiftMat.Mat::at<double>(1, 0) = 0;
	shiftMat.Mat::at<double>(1, 2) = 0;

	Mat dst;
	panoramic = panor.imSet[0];

	for (int i = 0; i < panor.imSet.size() - 1; i++) {

		shiftMat.cv::Mat::at<double>(0, 2) = -totalMeanDist[i];
		warpAffine(panor.imSet[(int)i + 1], dst, shiftMat, Size(panor.imSet[(int)i + 1].cols - totalMeanDist[i], panor.imSet[i + 1].rows), INTER_CUBIC, BORDER_CONSTANT, Scalar());
		hconcat(panoramic, dst, panoramic);

	}

	cout << "SHOWING PANORAMIC IN 3... 2... 1...\n" << endl;
	cv::namedWindow("panoramic");
	imshow("panoramic", panoramic);
	cout << "[Press any key to continue]\n\n";
	cv::waitKey(0);

}

//___________________________________________________________________________________________________________





//4._________________________________________________________________________________________________________

void PanoramicImage::findDistance(PanoramicImage panor, vector<vector<float>> & totalDistance, vector<vector<KeyPoint>> & totalKPoints, vector<vector<DMatch>> & totalInliersGoodMatches) {

	Point2f point1, point2;
	float dist;
	vector<float> distance;

	for (int i = 0; i < totalKPoints.size() - 1; i++)
	{
		for (int j = 0; j < totalInliersGoodMatches[i].size(); j++)
		{
			point1 = totalKPoints[i][totalInliersGoodMatches[i][j].queryIdx].pt;
			point2 = totalKPoints[i + 1][totalInliersGoodMatches[i][j].trainIdx].pt;
			dist = panor.imSet[i].cols - point1.x + point2.x;
			distance.push_back(dist);
		}
		totalDistance.push_back(distance);
		distance.clear();
	}
}

void PanoramicImage::showAndSavePanoramic(PanoramicImage panor, double ratio, string dstPathName)
{

	vector<vector<KeyPoint>> totalKPoints; /// For extraction of ORB features.
	vector<Mat> totalDescriptors;
	vector<vector<DMatch>> totalMatches; /// For the matcher	
	vector<vector<DMatch>> totalRefinedMatches; /// For refinement of the matches	
	vector<vector<DMatch>> totalInliersGoodMatches; /// For inliers retrivement	
	vector<float> totalMeanDist; /// For merging

	// Project the images on a cylinder surface
	PanoramicImage::cyProj(panor);

	// Extract the ORB features from the images
	PanoramicImage::orbFeaturesExtractor(panor, totalKPoints, totalDescriptors);

	// Compute the match between the different features of each (consecutive) couple of images
	PanoramicImage::matcher(panor, totalDescriptors, totalMatches);

	// Refine the matches found
	PanoramicImage::matchesRefiner(totalMatches, totalRefinedMatches, ratio);

	// Find the set of inliers
	PanoramicImage::inliersRetriever(totalKPoints, totalRefinedMatches, totalInliersGoodMatches);

	// Find the pixel distance
	vector<vector<float>> totalDistance;
	PanoramicImage::findDistance(panor, totalDistance, totalKPoints, totalInliersGoodMatches);

	// Merge images 
	Mat panoramic;
	PanoramicImage::mergeImg(panor, panoramic, totalMeanDist, totalDistance, totalInliersGoodMatches);

	imwrite(dstPathName, panoramic);
	cout << "PANORAMIC SAVED IN: " << dstPathName << endl;
}

//___________________________________________________________________________________________________________
