#include "panoramic_utils.h"
#include "PanoramicImage.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>    
#include <algorithm>    // std::sort
#include <vector> 

// Constructor
PanoramicImage::PanoramicImage(const std::vector<cv::Mat> imageSet, const double focalLength, const double verticalFOV)
{
	data = imageSet;				//vector of images
	focal_length = focalLength;     //focal length
	vFOV = verticalFOV;				// vertical field of view
}

//Load images function 
 void PanoramicImage::loadImages(cv::String path, PanoramicImage &panor)
{
	cv::String p = path;
	std::vector<cv::String> filenames;
	
	cv::glob(p, filenames);
	for (int i = 0; i < filenames.size(); i++)
	{
		cv::Mat im = cv::imread(filenames[i]);
		panor.data.push_back(im);
	}
}

//Function that project the images on a cylinder surface 
void PanoramicImage::cylProj(PanoramicImage& panor) 
{
	for (int i = 0; i < panor.data.size(); i++) 
	{
		PanoramicUtils::cylindricalProj(panor.data[i], panor.vFOV / 2);
	}
}

//Function that extract the keypoints from the images 
void PanoramicImage::orbExtract(PanoramicImage panor, std::vector<std::vector<cv::KeyPoint>>& allKPoints, std::vector<cv::Mat>& allDescriptors) 
{
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	for (int i = 0; i < panor.data.size(); i++) 
	{
		cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(); //Initiate ORB detector	
		detector->detect(panor.data[i], keypoints);
		allKPoints.push_back(keypoints);
		
		cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create(); //Initiate ORB extractor
		extractor->compute(panor.data[i], keypoints, descriptors);
		allDescriptors.push_back(descriptors);
	}

}

//Function that compute the match between the different features of each (consecutive) couple of images
void PanoramicImage::matcher(PanoramicImage panor, std::vector<cv::Mat>& allDescriptors, std::vector<std::vector<cv::DMatch>>& allMatches) 
{
	std::vector<cv::DMatch> matches;
	cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

	for (int i = 0; i < panor.data.size()-1; i++)
	{
		matcher->match(allDescriptors[i], allDescriptors[i + 1], matches);
		//std::sort(matches.begin(),matches.end());  //minimum distance is the first element
		allMatches.push_back(matches);
	}
}

//Function that refine the matches
void PanoramicImage::matchesRefiner(std::vector<std::vector<cv::DMatch>>& allMatches, std::vector<std::vector<cv::DMatch>>& allRefinedMatches, double ratio) 
{	
	std::vector<float> minDistances; //contains the minDist between the matches of (i) and (i+1) images.

	//Compute the minimum distance among matches for each couple of images
	for (int i = 0; i < allMatches.size(); i++)
	{
		std::sort(allMatches[i].begin(), allMatches[i].end());  //minimum distance is the first element at index 0
		minDistances.push_back(allMatches[i][0].distance);
	}

	//Refine using the ratio
	if (ratio == 0) { ratio = 3; }
	for (int i = 0; i < allMatches.size(); i++) 
	{
		std::vector<cv::DMatch> refinedMatches;
		for (int j = 0; j < allMatches[i].size(); j++) 
			if (allMatches[i][j].distance <= minDistances[i] * ratio) 
				refinedMatches.push_back(allMatches[i][j]);

		allRefinedMatches.push_back(refinedMatches);
	}
}

//Function that find the translation between the images
void PanoramicImage::findTranslImg(std::vector<std::vector<cv::KeyPoint>>& allKPoints, std::vector<std::vector<cv::DMatch>>& allRefinedMatches, std::vector<std::vector<cv::DMatch>>& allInliersGoodMatches)
{
	//Parameter for findHomography() function
	cv::Mat hmask;
	std::vector<cv::Mat> allHmask;

	//Fill totalHmask
	for (int i = 0; i < allKPoints.size()-1; i++) 
	{
		std::vector<cv::Point2f> points1, points2;
		for (int j = 0; j < allRefinedMatches[i].size(); j++)
		{
			points1.push_back(allKPoints[i][allRefinedMatches[i][j].queryIdx].pt);
			points2.push_back(allKPoints[i + 1][allRefinedMatches[i][j].trainIdx].pt);
		}

		findHomography(points1, points2, hmask, cv::RANSAC);
		allHmask.push_back(hmask);
	}

	//Fill allInliersGoodMatches
	for (int i = 0; i < allRefinedMatches.size(); i++)
	{
		std::vector<cv::DMatch> inliersGoodMatches; // fill the output vector allInliersGoodMatches
		for (int j = 0; j < allRefinedMatches[i].size(); j++)
		{
			if ((int)allHmask[i].Mat::at<uchar>(j, 0)) {inliersGoodMatches.push_back(allRefinedMatches[i][j]);}
		}
		allInliersGoodMatches.push_back(inliersGoodMatches);
	}
}


void PanoramicImage::mergeImg(PanoramicImage panor, cv::Mat& panoramic, std::vector<float>& allMeanDist, std::vector<std::vector<float>>& allDistance, std::vector<std::vector<cv::DMatch>>& allInliersGoodMatches)
{
	//Compute the mean distance between a couples of images
	float dist;
	float numInliers;

	for (int i = 0; i < allInliersGoodMatches.size(); i++)
	{
		dist = 0;
		numInliers = allInliersGoodMatches[i].size();

		for (int j = 0; j < numInliers; j++) 
		{
			dist = dist + allDistance[i][j];
		}

		dist = dist / numInliers;
		allMeanDist.push_back(dist);
	}

	//Apply translation of mean distance value
	cv::Mat shiftMat(2, 3, CV_64F, cv::Scalar(0.0));

	shiftMat.Mat::at<double>(0, 0) = 1;
	shiftMat.Mat::at<double>(1, 1) = 1;
	shiftMat.Mat::at<double>(0, 1) = 0;
	shiftMat.Mat::at<double>(1, 0) = 0;
	shiftMat.Mat::at<double>(1, 2) = 0;

	cv::Mat dst;
	panoramic = panor.data[0];

	for (int i = 0; i < panor.data.size() - 1; i++) 
	{
		shiftMat.cv::Mat::at<double>(0, 2) = -allMeanDist[i];
		warpAffine(panor.data[i + 1], dst, shiftMat, cv::Size(panor.data[i + 1].cols - allMeanDist[i], panor.data[i + 1].rows), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar());
		hconcat(panoramic, dst, panoramic);
	}

	cv::namedWindow("Panoramic");
	imshow("Panoramic", panoramic);
	cv::waitKey(0);
}

void PanoramicImage::findDistance(PanoramicImage pano, std::vector<std::vector<float>>& allDistance, std::vector<std::vector<cv::KeyPoint>>& allKPoints, std::vector<std::vector<cv::DMatch>>& allInliersGoodMatches)
{
	cv::Point2f point1, point2;
	float dist;


	for (int i = 0; i < allKPoints.size() - 1; i++)
	{
		std::vector<float> distance;
		for (int j = 0; j < allInliersGoodMatches[i].size(); j++)
		{
			point1 = allKPoints[i][allInliersGoodMatches[i][j].queryIdx].pt;
			point2 = allKPoints[i + 1][allInliersGoodMatches[i][j].trainIdx].pt;
			dist = pano.data[i].cols - point1.x + point2.x;
			distance.push_back(dist);
		}
		allDistance.push_back(distance);
	}
}

void PanoramicImage::showPanoramic(PanoramicImage panor, double ratio, cv::String dstPathName)
{
	// For image cylindrical projection.
	std::vector<std::vector<cv::KeyPoint>> totalKPoints; // For extraction of ORB features.
	std::vector<cv::Mat> totalDescriptors;
	std::vector<std::vector<cv::DMatch>> totalMatches; // For the matcher	
	std::vector<std::vector<cv::DMatch>> totalRefinedMatches; // For refinement of the matches	
	std::vector<std::vector<cv::DMatch>> totalInliersGoodMatches; // For inliers retrivement	
	std::vector<float> totalMeanDist; // For merging

	// Project the images on a cylinder surface
	PanoramicImage::cylProj(panor);

	// Extract the ORB features from the images
	PanoramicImage::orbExtract(panor, totalKPoints, totalDescriptors);

	// Compute the match between the different features of each (consecutive) couple of images
	PanoramicImage::matcher(panor, totalDescriptors, totalMatches);

	// Refine the matches found
	PanoramicImage::matchesRefiner(totalMatches, totalRefinedMatches, ratio);

	// Find the set of inliers
	PanoramicImage::findTranslImg(totalKPoints, totalRefinedMatches, totalInliersGoodMatches);

	// Find the pixel distance
	std::vector<std::vector<float>> totalDistance;
	PanoramicImage::findDistance(panor, totalDistance, totalKPoints, totalInliersGoodMatches);

	// Merge images 
	cv::Mat panoramic;
	PanoramicImage::mergeImg(panor, panoramic, totalMeanDist, totalDistance, totalInliersGoodMatches);

	imwrite(dstPathName, panoramic);
	std::cout << "Panoramic image saved in: " << dstPathName << std::endl;
	cv::waitKey(0);
}

//function to drawKeyPoints
void PanoramicImage::drawkPoints(PanoramicImage panor)
{
	std::vector<std::vector<cv::KeyPoint>> totalKPoints; 
	std::vector<cv::Mat> totalDescriptors;

	// Project the images on a cylinder surface
	PanoramicImage::cylProj(panor);
	// Extract the ORB features from the images
	PanoramicImage::orbExtract(panor, totalKPoints, totalDescriptors);
	cv::Mat out;

	for (int i = 0; i < panor.data.size(); i++)
	{
		cv::drawKeypoints(panor.data[i], totalKPoints[i], out);
		cv::namedWindow("ORB result");
		cv::imshow("ORB result", out);
		cv::waitKey(0);
	}
}