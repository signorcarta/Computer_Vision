#include "ObjectDetection.h"
#include <memory>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>


// Constructor
ObjectDetection::ObjectDetection( std::string path, std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors ) :
imageTrain( cv::imread( path ) ), keypoints( keypoints ), descriptors( descriptors )
{
   //imageTrain = cv::imread( path );
   if ( !imageTrain.data )
   {
      std::cout << "--(!) Error reading image" << std::endl;
   }
}



// Computing the training keypoints and descriptors from the object train image
void ObjectDetection::SiftFeaturesExtractor( ObjectDetection& obj )
{
   cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
   cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
   detector->detect( obj.imageTrain, obj.keypoints );
   extractor->compute( obj.imageTrain, obj.keypoints, obj.descriptors );
}



// Computing the matches between the training and testing descriptors.
void ObjectDetection::matcher( ObjectDetection& train, ObjectDetection& scene, std::vector<cv::DMatch>& matches )
{
   cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create( cv::NORM_L1, false );
   matcher->match( train.descriptors, scene.descriptors, matches );
   std::cout << "MATCHES FOUND" << std::endl;
}



// Refine the matches found by ObjectDetection::matcher by considering
// only the matches with distance less than ratio times the whole minimum
// distance value between the matches.
void ObjectDetection::matchesRefiner( std::vector<cv::DMatch>& matches,
                                     std::vector<cv::DMatch>& goodMatches,
                                     double ratio )
{
   if ( ratio<1 )
   {
      std::cout << "--(!) Ratio value must be greater (or equal) than one!";
      return;
   }
   // Find the minimum distance between the matches
   float minDist = matches[0].distance; // candidate to be the minimum
   for ( int i=1; i<matches.size(); i++ )
   {
      if ( matches[i].distance < minDist )
         minDist = matches[i].distance;
   }
   // Refinement
   for ( int i=0; i<matches.size(); i++ )
   {
      if ( matches[i].distance <= ratio*minDist )
         goodMatches.push_back( matches[i] );
   }
}

void ObjectDetection::inliersRetriver( ObjectDetection& train,
                                       ObjectDetection& scene,
                                       const std::vector<cv::DMatch>& goodMatches,
                                       std::vector<cv::DMatch>& inliersGoodMatches )
{
   cv::Mat hmask;
   std::vector<cv::Point2f> points1, points2;
   for ( int i=0; i<goodMatches.size(); i++ )
   {
		// Get the keypoints from the good matches
		points1.push_back(train.keypoints[ goodMatches[i].queryIdx ].pt);
		points2.push_back(scene.keypoints[ goodMatches[i].trainIdx ].pt);
	}
   cv::Mat H = cv::findHomography( points1, points2, hmask, CV_RANSAC );
   for ( int i=0; i<goodMatches.size(); i++ )
   {
      if ( (int)hmask.cv::Mat::at<uchar>(i,0) )
         inliersGoodMatches.push_back( goodMatches[i] );
   }
   std::cout << "INLIERS RETRIVED" << std::endl;
}

// It searches for the right position, orientation and scale of the object in the scene based on the good_matches
void ObjectDetection::localizeInImage( ObjectDetection& train,
                                       ObjectDetection& scene,
                                       cv::Mat& img_matches,
                                       const std::vector<cv::DMatch>& goodMatches )
{
   // Localize the object
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scn;
   for ( int i=0; i<goodMatches.size(); i++ )
   {
		// Get the keypoints from the good matches
		obj.push_back(train.keypoints[ goodMatches[i].queryIdx ].pt);
		scn.push_back(scene.keypoints[ goodMatches[i].trainIdx ].pt);
	}
   cv::Mat H = cv::findHomography( obj, scn, CV_RANSAC );

   // Get the corners from the image ( the object to be "detected" )
	std::vector<cv::Point2f> obj_corners( 4 );
	obj_corners[0] = cvPoint( 0, 0 );
	obj_corners[1] = cvPoint( train.imageTrain.cols, 0 );
	obj_corners[2] = cvPoint( train.imageTrain.cols, train.imageTrain.rows );
	obj_corners[3] = cvPoint( 0, train.imageTrain.rows );
	std::vector<cv::Point2f> scene_corners( 4 );

   perspectiveTransform( obj_corners, scene_corners, H );

   // Draw lines between the corners (the mapped object in the scene )
	line( img_matches, scene_corners[0] + cv::Point2f(train.imageTrain.cols, 0),
		  scene_corners[1] + cv::Point2f(train.imageTrain.cols, 0),
		  cv::Scalar(255, 0, 0), 4 );
	line( img_matches, scene_corners[1] + cv::Point2f(train.imageTrain.cols, 0),
		  scene_corners[2] + cv::Point2f(train.imageTrain.cols, 0),
		  cv::Scalar(255, 0, 0), 4 );
	line( img_matches, scene_corners[2] + cv::Point2f(train.imageTrain.cols, 0),
		  scene_corners[3] + cv::Point2f(train.imageTrain.cols, 0),
		  cv::Scalar(255, 0, 0), 4 );
	line( img_matches, scene_corners[3] + cv::Point2f(train.imageTrain.cols, 0),
		  scene_corners[0] + cv::Point2f(train.imageTrain.cols, 0),
		  cv::Scalar(255, 0, 0), 4 );
}
