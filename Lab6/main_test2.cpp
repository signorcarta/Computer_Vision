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
   Mat descriptors1, descriptors2, descriptors3, descriptors4;
   Mat descriptorsScene1, descriptorsScene2, descriptorsScene3, descriptorsScene4;
   vector<KeyPoint> keypoints1, keypoints2, keypoints3, keypoints4;
   vector<KeyPoint> keypointsScene1, keypointsScene2, keypointsScene3, keypointsScene4;

   ObjectDetection obj1 = ObjectDetection( "../data/dataset2/obj1.png", keypoints1, descriptors1 );
   ObjectDetection obj2 = ObjectDetection( "../data/dataset2/obj2.png", keypoints2, descriptors2 );
   ObjectDetection obj3 = ObjectDetection( "../data/dataset2/obj3.png", keypoints3, descriptors3 );
   ObjectDetection obj4 = ObjectDetection( "../data/dataset2/obj4.png", keypoints4, descriptors4 );
   ObjectDetection scene1 = ObjectDetection( "../data/dataset2/scene1.png", keypointsScene1, descriptorsScene1 );
   ObjectDetection scene2 = ObjectDetection( "../data/dataset2/scene2.png", keypointsScene2, descriptorsScene2 );
   ObjectDetection scene3 = ObjectDetection( "../data/dataset2/scene3.png", keypointsScene3, descriptorsScene3 );
   ObjectDetection scene4 = ObjectDetection( "../data/dataset2/scene4.png", keypointsScene4, descriptorsScene4 );


   // Computing the training keypoints and descriptors from the object train image
   ObjectDetection::SiftFeaturesExtractor( obj1 );
   ObjectDetection::SiftFeaturesExtractor( obj2 );
   ObjectDetection::SiftFeaturesExtractor( obj3 );
   ObjectDetection::SiftFeaturesExtractor( obj4 );

   // Computing the testing keypointsScene and descriptorsScene from a scene image
   ObjectDetection::SiftFeaturesExtractor( scene1 );
   ObjectDetection::SiftFeaturesExtractor( scene2 );
   ObjectDetection::SiftFeaturesExtractor( scene3 );
   ObjectDetection::SiftFeaturesExtractor( scene4 );


   // Computing the matches between the training and testing descriptors
   Mat  img_matches11, img_matches12, img_matches13, img_matches14,
        img_matches21, img_matches22, img_matches23, img_matches24,
        img_matches31, img_matches32, img_matches33, img_matches34,
        img_matches41, img_matches42, img_matches43, img_matches44;

   vector<DMatch> matches11, matches12, matches13, matches14,
                  matches21, matches22, matches23, matches24,
                  matches31, matches32, matches33, matches34,
                  matches41, matches42, matches43, matches44;

   ObjectDetection::matcher( obj1, scene1, matches11 );
   ObjectDetection::matcher( obj1, scene2, matches12 );
   ObjectDetection::matcher( obj1, scene3, matches13 );
   ObjectDetection::matcher( obj1, scene4, matches14 );

   ObjectDetection::matcher( obj2, scene1, matches21 );
   ObjectDetection::matcher( obj2, scene2, matches22 );
   ObjectDetection::matcher( obj2, scene3, matches23 );
   ObjectDetection::matcher( obj2, scene4, matches24 );

   ObjectDetection::matcher( obj3, scene1, matches31 );
   ObjectDetection::matcher( obj3, scene2, matches32 );
   ObjectDetection::matcher( obj3, scene3, matches33 );
   ObjectDetection::matcher( obj3, scene4, matches34 );

   ObjectDetection::matcher( obj4, scene1, matches41 );
   ObjectDetection::matcher( obj4, scene2, matches42 );
   ObjectDetection::matcher( obj4, scene3, matches43 );
   ObjectDetection::matcher( obj4, scene4, matches44 );

   // Refine the matches found by ObjectDetection::matcher by considering
   // only the matches with distance less than ratio times the whole minimum
   // distance value between the matches.
   double ratio = 10;
   std::vector<cv::DMatch> goodMatches11, goodMatches12, goodMatches13, goodMatches14,
                           goodMatches21, goodMatches22, goodMatches23, goodMatches24,
                           goodMatches31, goodMatches32, goodMatches33, goodMatches34,
                           goodMatches41, goodMatches42, goodMatches43, goodMatches44;

   ObjectDetection::matchesRefiner( matches11, goodMatches11, ratio );
   ObjectDetection::matchesRefiner( matches12, goodMatches12, ratio );
   ObjectDetection::matchesRefiner( matches13, goodMatches13, ratio );
   ObjectDetection::matchesRefiner( matches14, goodMatches14, ratio );

   ObjectDetection::matchesRefiner( matches21, goodMatches21, ratio );
   ObjectDetection::matchesRefiner( matches22, goodMatches22, ratio );
   ObjectDetection::matchesRefiner( matches23, goodMatches23, ratio );
   ObjectDetection::matchesRefiner( matches24, goodMatches24, ratio );

   ObjectDetection::matchesRefiner( matches31, goodMatches31, ratio );
   ObjectDetection::matchesRefiner( matches32, goodMatches32, ratio );
   ObjectDetection::matchesRefiner( matches33, goodMatches33, ratio );
   ObjectDetection::matchesRefiner( matches34, goodMatches34, ratio );

   ObjectDetection::matchesRefiner( matches41, goodMatches41, ratio );
   ObjectDetection::matchesRefiner( matches42, goodMatches42, ratio );
   ObjectDetection::matchesRefiner( matches43, goodMatches43, ratio );
   ObjectDetection::matchesRefiner( matches44, goodMatches44, ratio );

   // Inliers retrivement
   vector<DMatch> inliersGoodMatches11, inliersGoodMatches12, inliersGoodMatches13, inliersGoodMatches14,
                  inliersGoodMatches21, inliersGoodMatches22, inliersGoodMatches23, inliersGoodMatches24,
                  inliersGoodMatches31, inliersGoodMatches32, inliersGoodMatches33, inliersGoodMatches34,
                  inliersGoodMatches41, inliersGoodMatches42, inliersGoodMatches43, inliersGoodMatches44;

   ObjectDetection::inliersRetriver( obj1, scene1, goodMatches11, inliersGoodMatches11 );
   ObjectDetection::inliersRetriver( obj1, scene2, goodMatches12, inliersGoodMatches12 );
   ObjectDetection::inliersRetriver( obj1, scene3, goodMatches13, inliersGoodMatches13 );
   ObjectDetection::inliersRetriver( obj1, scene4, goodMatches14, inliersGoodMatches14 );

   ObjectDetection::inliersRetriver( obj2, scene1, goodMatches21, inliersGoodMatches21 );
   ObjectDetection::inliersRetriver( obj2, scene2, goodMatches22, inliersGoodMatches22 );
   ObjectDetection::inliersRetriver( obj2, scene3, goodMatches23, inliersGoodMatches23 );
   ObjectDetection::inliersRetriver( obj2, scene4, goodMatches24, inliersGoodMatches24 );

   ObjectDetection::inliersRetriver( obj3, scene1, goodMatches31, inliersGoodMatches31 );
   ObjectDetection::inliersRetriver( obj3, scene2, goodMatches32, inliersGoodMatches32 );
   ObjectDetection::inliersRetriver( obj3, scene3, goodMatches33, inliersGoodMatches33 );
   ObjectDetection::inliersRetriver( obj3, scene4, goodMatches34, inliersGoodMatches34 );

   ObjectDetection::inliersRetriver( obj4, scene1, goodMatches41, inliersGoodMatches41 );
   ObjectDetection::inliersRetriver( obj4, scene2, goodMatches42, inliersGoodMatches42 );
   ObjectDetection::inliersRetriver( obj4, scene3, goodMatches43, inliersGoodMatches43 );
   ObjectDetection::inliersRetriver( obj4, scene4, goodMatches44, inliersGoodMatches44 );

   drawMatches( obj1.imageTrain, obj1.keypoints, scene1.imageTrain, scene1.keypoints, inliersGoodMatches11, img_matches11,
                Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj1.imageTrain, obj1.keypoints, scene2.imageTrain, scene2.keypoints, inliersGoodMatches12, img_matches12,
                Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj1.imageTrain, obj1.keypoints, scene3.imageTrain, scene3.keypoints, inliersGoodMatches13, img_matches13,
                Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj1.imageTrain, obj1.keypoints, scene4.imageTrain, scene4.keypoints, inliersGoodMatches14, img_matches14,
                Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj2.imageTrain, obj2.keypoints, scene1.imageTrain, scene1.keypoints, inliersGoodMatches21, img_matches21,
                             Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj2.imageTrain, obj2.keypoints, scene2.imageTrain, scene2.keypoints, inliersGoodMatches22, img_matches22,
                             Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj2.imageTrain, obj2.keypoints, scene3.imageTrain, scene3.keypoints, inliersGoodMatches23, img_matches23,
                             Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj2.imageTrain, obj2.keypoints, scene4.imageTrain, scene4.keypoints, inliersGoodMatches24, img_matches24,
                             Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj3.imageTrain, obj3.keypoints, scene1.imageTrain, scene1.keypoints, inliersGoodMatches31, img_matches31,
                                          Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj3.imageTrain, obj3.keypoints, scene2.imageTrain, scene2.keypoints, inliersGoodMatches32, img_matches32,
                                          Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj3.imageTrain, obj3.keypoints, scene3.imageTrain, scene3.keypoints, inliersGoodMatches33, img_matches33,
                                          Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj3.imageTrain, obj3.keypoints, scene4.imageTrain, scene4.keypoints, inliersGoodMatches34, img_matches34,
                                          Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj4.imageTrain, obj4.keypoints, scene1.imageTrain, scene1.keypoints, inliersGoodMatches41, img_matches41,
                                                       Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj4.imageTrain, obj4.keypoints, scene2.imageTrain, scene2.keypoints, inliersGoodMatches42, img_matches42,
                                                       Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj4.imageTrain, obj4.keypoints, scene3.imageTrain, scene3.keypoints, inliersGoodMatches43, img_matches43,
                                                       Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

   drawMatches( obj4.imageTrain, obj4.keypoints, scene4.imageTrain, scene4.keypoints, inliersGoodMatches44, img_matches44,
                                                       Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


   // It searches for the right position, orientation and scale of the object
   // in the scene based on the good_matches
   ObjectDetection::localizeInImage( obj1, scene1, img_matches11, goodMatches11 );
   ObjectDetection::localizeInImage( obj1, scene2, img_matches12, goodMatches12 );
   ObjectDetection::localizeInImage( obj1, scene3, img_matches13, goodMatches13 );
   ObjectDetection::localizeInImage( obj1, scene4, img_matches14, goodMatches14 );

   ObjectDetection::localizeInImage( obj2, scene1, img_matches21, goodMatches21 );
   ObjectDetection::localizeInImage( obj2, scene2, img_matches22, goodMatches22 );
   ObjectDetection::localizeInImage( obj2, scene3, img_matches23, goodMatches23 );
   ObjectDetection::localizeInImage( obj2, scene4, img_matches24, goodMatches24 );

   ObjectDetection::localizeInImage( obj3, scene1, img_matches31, goodMatches31 );
   ObjectDetection::localizeInImage( obj3, scene2, img_matches32, goodMatches32 );
   ObjectDetection::localizeInImage( obj3, scene2, img_matches33, goodMatches33 );
   ObjectDetection::localizeInImage( obj3, scene4, img_matches34, goodMatches34 );

   ObjectDetection::localizeInImage( obj4, scene1, img_matches41, goodMatches41 );
   ObjectDetection::localizeInImage( obj4, scene2, img_matches42, goodMatches42 );
   ObjectDetection::localizeInImage( obj4, scene3, img_matches43, goodMatches43 );
   ObjectDetection::localizeInImage( obj4, scene4, img_matches44, goodMatches44 );

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

   cv::resize( img_matches41, img_matches41, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object4 detection in scene1", img_matches41);
   waitKey(0);
   destroyWindow("Inliers good matches & object4 detection in scene1");

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

   cv::resize( img_matches42, img_matches42, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object4 detection in scene2", img_matches42);
   waitKey(0);
   destroyWindow("Inliers good matches & object4 detection in scene2");

   cv::resize( img_matches13, img_matches13, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object1 detection in scene3", img_matches13);
	waitKey(0);
   destroyWindow("Inliers good matches & object1 detection in scene3");

   cv::resize( img_matches23, img_matches23, cv::Size(), 0.5, 0.5);
   imwrite("../obj2Scene3.png", img_matches23);
   imshow("Inliers good matches & object2 detection in scene3", img_matches23);
   waitKey(0);
   destroyWindow("Inliers good matches & object2 detection in scene3");

   cv::resize( img_matches33, img_matches33, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object3 detection in scene3", img_matches33);
   waitKey(0);
   destroyWindow("Inliers good matches & object3 detection in scene3");

   cv::resize( img_matches43, img_matches43, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object4 detection in scene3", img_matches43);
   waitKey(0);
   destroyWindow("Inliers good matches & object4 detection in scene3");

   cv::resize( img_matches14, img_matches14, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object1 detection in scene4", img_matches14);
	waitKey(0);
   destroyWindow("Inliers good matches & object1 detection in scene4");

   cv::resize( img_matches24, img_matches24, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object2 detection in scene4", img_matches24);
   waitKey(0);
   destroyWindow("Inliers good matches & object2 detection in scene4");

   cv::resize( img_matches34, img_matches34, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object3 detection in scene4", img_matches34);
   waitKey(0);
   destroyWindow("Inliers good matches & object3 detection in scene4");

   cv::resize( img_matches44, img_matches44, cv::Size(), 0.5, 0.5);
   imshow("Inliers good matches & object4 detection in scene4", img_matches44);
   waitKey(0);
   destroyWindow("Inliers good matches & object4 detection in scene4");

   return 0;
}
