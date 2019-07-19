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
   Mat descriptors1, descriptors2, descriptors3, descriptorsScene1;
   vector<KeyPoint> keypoints1, keypoints2, keypoints3, keypointsScene1;

   ObjectDetection obj1 = ObjectDetection( "../data/dataset1/obj1.png", keypoints1, descriptors1 );
   ObjectDetection obj2 = ObjectDetection( "../data/dataset1/obj2.png", keypoints2, descriptors2 );
   ObjectDetection obj3 = ObjectDetection( "../data/dataset1/obj3.png", keypoints3, descriptors3 );
   ObjectDetection scene1 = ObjectDetection( "../data/dataset1/scene1.png", keypointsScene1, descriptorsScene1 );


   // Computing the training keypoints and descriptors from the object train image
   ObjectDetection::SiftFeaturesExtractor( obj1 );
   ObjectDetection::SiftFeaturesExtractor( obj2 );
   ObjectDetection::SiftFeaturesExtractor( obj3 );

   // Computing the testing keypointsScene1 and descriptorsScene1 from a scene image
   ObjectDetection::SiftFeaturesExtractor( scene1 );

   // Computing the matches between the training and testing descriptors
   cv::Mat img_matches1, img_matches2, img_matches3;
   std::vector<cv::DMatch> matches1, matches2, matches3;
   ObjectDetection::matcher( obj1, scene1, matches1 );
   ObjectDetection::matcher( obj2, scene1, matches2 );
   ObjectDetection::matcher( obj3, scene1, matches3 );

   // Refine the matches found by ObjectDetection::matcher by considering
   // only the matches with distance less than ratio times the whole minimum
   // distance value between the matches.
   double ratio = 3;
   std::vector<cv::DMatch> goodMatches1, goodMatches2, goodMatches3;
   ObjectDetection::matchesRefiner( matches1, goodMatches1, ratio );
   ObjectDetection::matchesRefiner( matches2, goodMatches2, ratio );
   ObjectDetection::matchesRefiner( matches3, goodMatches3, ratio );

   // Inliers retrivement
   vector<DMatch> inliersGoodMatches1, inliersGoodMatches2, inliersGoodMatches3;
   ObjectDetection::inliersRetriver( obj1, scene1, goodMatches1, inliersGoodMatches1 );
   ObjectDetection::inliersRetriver( obj2, scene1, goodMatches2, inliersGoodMatches2 );
   ObjectDetection::inliersRetriver( obj3, scene1, goodMatches3, inliersGoodMatches3 );


   drawMatches( obj1.imageTrain, obj1.keypoints, scene1.imageTrain, scene1.keypoints, inliersGoodMatches1, img_matches1,
                Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj2.imageTrain, obj2.keypoints, scene1.imageTrain, scene1.keypoints, inliersGoodMatches2, img_matches2,
                Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj3.imageTrain, obj3.keypoints, scene1.imageTrain, scene1.keypoints, inliersGoodMatches3, img_matches3,
                Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   // It searches for the right position, orientation and scale of the object
   // in the scene based on the good_matches
   ObjectDetection::localizeInImage( obj1, scene1, img_matches1, goodMatches1 );
   ObjectDetection::localizeInImage( obj2, scene1, img_matches2, goodMatches2 );
   ObjectDetection::localizeInImage( obj3, scene1, img_matches3, goodMatches3 );

   // Show matches
   imshow("Inliers good matches & object1 detection", img_matches1);
	waitKey(0);

   imshow("Inliers good matches & object2 detection", img_matches2);
   waitKey(0);

   imshow("Inliers good matches & object3 detection", img_matches3);
   waitKey(0);

   return 0;
}
