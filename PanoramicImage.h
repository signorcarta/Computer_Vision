#ifndef LAB5_PANORAMIC_IMAGE_H
#define LAB5_PANORAMIC_IMAGE_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "panoramic_utils.h"

class PanoramicImage : PanoramicUtils {

    public:

		//Constructor
		PanoramicImage(const std::vector<cv::Mat> imageSet, const double focalLength, const double verticalFOV);
	

		//Function declaration

		//load images 
		static  void loadImages(cv::String path, PanoramicImage &panor);

		//Project the images on a cylinder surface
		static void cylProj(PanoramicImage &panor);

		//Extract ORB features from the image
		static void orbExtract(PanoramicImage panor, std::vector<std::vector<cv::KeyPoint>>& allKPoints, std::vector<cv::Mat>& allDescriptors);

		//Draw and show the key points detected from ORB
		static void drawkPoints(PanoramicImage panor);

		//Compute the match between the different features of each (consecutive) couple of images 
		static void matcher(PanoramicImage panor, std::vector<cv::Mat>& allDescriptors, std::vector<std::vector<cv::DMatch>>& allMatches);

		//Refine the matches
		static void matchesRefiner(std::vector<std::vector<cv::DMatch>>& allMatches, std::vector<std::vector<cv::DMatch>>& allRefinedMatches, double ratio);

		//Find the translation between the images
		static void findTranslImg(std::vector<std::vector<cv::KeyPoint>>& allKPoints, std::vector<std::vector<cv::DMatch>>& allRefinedMatches, std::vector<std::vector<cv::DMatch>>& allInliersGoodMatches);

		//Compute the final panorama
		static void mergeImg(PanoramicImage panor, cv::Mat& panoramic, std::vector<float>& allMeanDist, std::vector<std::vector<float>>& allDistance, std::vector<std::vector<cv::DMatch>>& allInliersGoodMatches);
		
		//
		static void findDistance(PanoramicImage pano, std::vector<std::vector<float>>& allDistance, std::vector<std::vector<cv::KeyPoint>>& allKPoints, std::vector<std::vector<cv::DMatch>>& allInliersGoodMatches);

		//Show and Save result
		static void showPanoramic(PanoramicImage panor, double ratio, cv::String dstPathName);

//Variables
private:
	std::vector<cv::Mat> data;
	double focal_length; //[mm] focal length
	double vFOV; //[degrees] vertical field of view
};
#endif //LAB5_PANORAMIC_IMAGE_H