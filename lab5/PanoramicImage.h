#ifndef LAB5__PANORAMIC__IMAGE__H
#define LAB5__PANORAMIC__IMAGE__H

#include "opencv2/opencv.hpp" 
#include <opencv2/core.hpp>
#include <vector>
#include "PanoramicUtils.h"

using namespace std;
using namespace cv;

class PanoramicImage : PanoramicUtils {
	
public:

	///Variables______________________________________________
	vector<Mat> imSet;
	double focal_len; // focal length [mm]
	double vFOV; // vertical field of view [degrees]
	///_______________________________________________________



	///Constructor____________________________________________

	PanoramicImage(const vector<Mat> imageSet, const double focalLength, const double verticalFOV);

	///_______________________________________________________


	///Functions declaration__________________________________

	//Load images
	static vector<Mat> loadImages(string path, 
		                          int& numImg); 

	//Show and Save result
	static void showAndSavePanoramic(PanoramicImage panor, 
		                             double ratio, 
		                             string dstPathName);

	//Project the images on a cylinder surface
	static void cyProj(PanoramicImage& panor);

	//Extract ORB features from the image
	static void orbFeaturesExtractor(PanoramicImage panor, 
									 vector<vector<KeyPoint>>& totalKPoints, 
		                             vector<Mat>& totalDescriptors);
	
	//Compute the match between the different features of each (consecutive) couple of images 
	static void matcher(PanoramicImage panor, 
		                vector<Mat>& totalDescriptors, 
		                vector<vector<DMatch>>& totalMatches);

	//Refine the matches
	static void matchesRefiner(vector<vector<DMatch>>& totalMatches, 
		                       vector<vector<DMatch>>& totalRefinedMatches, 
		                       double ratio);

	//Find the translation between the images
	static void inliersRetriever(vector<vector<KeyPoint>>& totalKPoints, 
		                         vector<vector<DMatch>>& totalRefinedMatches, 
		                         vector<vector<DMatch>>& totalInliersGoodMatches);

	//Compute the final panorama
	static void mergeImg(PanoramicImage panor,
						 Mat& panoramic,
						 vector<float>& totalMeanDist,
						 vector<vector<float>>& totalDistance,
						 vector<vector<DMatch>>& totalInliersGoodMatches);

	///____________________________________________________

};
#endif
