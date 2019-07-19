#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "ObjectDetection.h"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

int main( int argc, char** argv )
{
   // Import train and scene image
   Mat descriptors1, descriptors2, descriptors3;
   Mat descriptorsScene1, descriptorsScene2;
   vector<KeyPoint> keypoints1, keypoints2, keypoints3;
   vector<KeyPoint> keypointsScene1, keypointsScene2;

   ObjectDetection obj1 = ObjectDetection( "../data/dataset3/obj1.png", keypoints1, descriptors1 );
   ObjectDetection obj2 = ObjectDetection( "../data/dataset3/obj2.png", keypoints2, descriptors2 );
   ObjectDetection obj3 = ObjectDetection( "../data/dataset3/obj3.png", keypoints3, descriptors3 );
   ObjectDetection scene1 = ObjectDetection( "../data/dataset3/scene1.png", keypointsScene1, descriptorsScene1 );
   ObjectDetection scene2 = ObjectDetection( "../data/dataset3/scene2.png", keypointsScene2, descriptorsScene2 );


   // Computing the training keypoints and descriptors from the object train image
   ObjectDetection::SiftFeaturesExtractor( obj1 );
   ObjectDetection::SiftFeaturesExtractor( obj2 );
   ObjectDetection::SiftFeaturesExtractor( obj3 );

   // Computing the testing keypointsScene1 and descriptorsScene1 from a scene image
   ObjectDetection::SiftFeaturesExtractor( scene1 );
   ObjectDetection::SiftFeaturesExtractor( scene2 );

   // Computing the matches between the training and testing descriptors
   Mat  img_matches11, img_matches12,
        img_matches21, img_matches22,
        img_matches31, img_matches32;

   vector<DMatch> matches11, matches12,
                  matches21, matches22,
                  matches31, matches32;

   ObjectDetection::matcher( obj1, scene1, matches11 );
   ObjectDetection::matcher( obj2, scene1, matches21 );
   ObjectDetection::matcher( obj3, scene1, matches31 );

   ObjectDetection::matcher( obj1, scene2, matches12 );
   ObjectDetection::matcher( obj2, scene2, matches22 );
   ObjectDetection::matcher( obj3, scene2, matches32 );

   // Refine the matches found by ObjectDetection::matcher by considering
   // only the matches with distance less than ratio times the whole minimum
   // distance value between the matches.
   double ratio = 10;
   std::vector<cv::DMatch> goodMatches11, goodMatches12,
                           goodMatches21, goodMatches22,
                           goodMatches31, goodMatches32;

   ObjectDetection::matchesRefiner( matches11, goodMatches11, ratio );
   ObjectDetection::matchesRefiner( matches21, goodMatches21, ratio );
   ObjectDetection::matchesRefiner( matches31, goodMatches31, ratio );
   ObjectDetection::matchesRefiner( matches12, goodMatches12, ratio );
   ObjectDetection::matchesRefiner( matches22, goodMatches22, ratio );
   ObjectDetection::matchesRefiner( matches32, goodMatches32, ratio );

   // Inliers retrivement
   vector<DMatch> inliersGoodMatches11, inliersGoodMatches12,
                  inliersGoodMatches21, inliersGoodMatches22,
                  inliersGoodMatches31, inliersGoodMatches32;

   ObjectDetection::inliersRetriver( obj1, scene1, goodMatches11, inliersGoodMatches11 );
   ObjectDetection::inliersRetriver( obj2, scene1, goodMatches21, inliersGoodMatches21 );
   ObjectDetection::inliersRetriver( obj3, scene1, goodMatches31, inliersGoodMatches31 );
   ObjectDetection::inliersRetriver( obj1, scene2, goodMatches12, inliersGoodMatches12 );
   ObjectDetection::inliersRetriver( obj2, scene2, goodMatches22, inliersGoodMatches22 );
   ObjectDetection::inliersRetriver( obj3, scene2, goodMatches32, inliersGoodMatches32 );

   drawMatches( obj1.imageTrain, obj1.keypoints, scene1.imageTrain, scene1.keypoints, inliersGoodMatches11, img_matches11,
                Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj2.imageTrain, obj2.keypoints, scene1.imageTrain, scene1.keypoints, inliersGoodMatches21, img_matches21,
                Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj3.imageTrain, obj3.keypoints, scene1.imageTrain, scene1.keypoints, inliersGoodMatches31, img_matches31,
                Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj1.imageTrain, obj1.keypoints, scene2.imageTrain, scene2.keypoints, inliersGoodMatches12, img_matches12,
                             Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj2.imageTrain, obj2.keypoints, scene2.imageTrain, scene2.keypoints, inliersGoodMatches22, img_matches22,
                             Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj3.imageTrain, obj3.keypoints, scene2.imageTrain, scene2.keypoints, inliersGoodMatches32, img_matches32,
                             Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   // It searches for the right position, orientation and scale of the object
   // in the scene based on the good_matches
   ObjectDetection::localizeInImage( obj1, scene1, img_matches11, goodMatches11 );
   ObjectDetection::localizeInImage( obj2, scene1, img_matches21, goodMatches21 );
   ObjectDetection::localizeInImage( obj3, scene1, img_matches31, goodMatches31 );
   ObjectDetection::localizeInImage( obj1, scene2, img_matches12, goodMatches12 );
   ObjectDetection::localizeInImage( obj2, scene2, img_matches22, goodMatches22 );
   ObjectDetection::localizeInImage( obj3, scene2, img_matches32, goodMatches32 );

   // Show matches
   cv::resize( img_matches11, img_matches11, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object1 detection in scene1", img_matches11);
	waitKey(0);
   destroyWindow("Inliers good matches & object1 detection in scene1");

   cv::resize( img_matches21, img_matches21, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object2 detection in scene1", img_matches21);
   waitKey(0);
   destroyWindow("Inliers good matches & object2 detection in scene1");

   cv::resize( img_matches31, img_matches31, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object3 detection in scene1", img_matches31);
   waitKey(0);
   destroyWindow("Inliers good matches & object3 detection in scene1");

   cv::resize( img_matches12, img_matches12, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object1 detection in scene2", img_matches12);
	waitKey(0);
   destroyWindow("Inliers good matches & object1 detection in scene2");

   cv::resize( img_matches22, img_matches22, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object2 detection in scene2", img_matches22);
   waitKey(0);
   destroyWindow("Inliers good matches & object2 detection in scene2");

   cv::resize( img_matches32, img_matches32, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object3 detection in scene2", img_matches32);
   waitKey(0);
   destroyWindow("Inliers good matches & object3 detection in scene2");



   return 0;
}
