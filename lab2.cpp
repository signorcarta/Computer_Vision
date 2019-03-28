#include <iostream>
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <vector>

#define IMG_NUM 57
#define SQ_SIZE 0.102 //meters
#define INNER_ROWS 5
#define INNER_COLS 6


int main(int argc, char** argv)
{

  // Size of the checkerboard inner pattern
  cv::Size patternSize = cv::Size( INNER_ROWS, INNER_COLS );

  // Half of the side length of the search window. For example, if
  // winSize=Size(5,5) , then a 5∗2+1×5∗2+1=11×11 search window is used.
  cv::Size winSize = cv::Size( 11, 11 );
  cv::Size ZeroZone = cv::Size( -1, -1 );
  cv::TermCriteria criteria = cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.00001 );
  cv::TermCriteria criteria_calib = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.00001);

  // Vector of detected corners coordinates of an image.
  std::vector<cv::Vec2f> corners;

  // Vector of x, y, z coordinates of inner corners of an image.
  std::vector<cv::Vec3f> objPoints;

  // Vector of objPoints of all images.
  std::vector< std::vector<cv::Vec3f> > objectPoints;

  // Vector of detected corners of all images.
  std::vector< std::vector<cv::Vec2f> > imagePoints;

  // Vector fot projected points during the computation of reprojection error
  std::vector<cv::Vec2f> prjPoints;

  // Initializing of cameraCalibration input.
  cv::Mat img_resized;
  cv::Size imageSize;
  cv::Mat grayImage;
  cv::Mat cameraMatrix;
  cv::Mat distCoeffs;
  std::vector<cv::Mat> rvecs, tvecs;

  // Initializing of total images vector (allocation memory for this container);
  cv::Mat total_imgs[IMG_NUM];

  // Char vector buffer (string length 35) and number name of image.
  int n = 1;
  char PATH_NAME [35];
  char DEST [35];

  // Prepare objPoints (each image has the same checkerboard pattern structure).
  for ( int c = 1; c<=INNER_COLS; c++ )
  {
    for ( int r = 1; r<=INNER_ROWS; r++ )
    {
      objPoints.push_back( cv::Vec3f( c*SQ_SIZE, r*SQ_SIZE, 0 ) );
    }
  }

  for ( int i = 0; i<IMG_NUM; i++ )
  {
    // Import images
    sprintf ( PATH_NAME, "../checkerboard_test/IMG_%d.png", n );
    total_imgs[i] = cv::imread( PATH_NAME );
    printf( "%s\n", PATH_NAME ) ;

    // Pattern detection
    bool patternfound = cv::findChessboardCorners( total_imgs[i], patternSize, corners );
    std::cout << ( patternfound!=0 ? "PATTERN FOUND" : "PATTERN NOT FOUND" ) << std::endl;
    if ( patternfound )
    {
      cv::cvtColor( total_imgs[i], grayImage, CV_BGR2GRAY );
      cv::cornerSubPix( grayImage, corners, cv::Size( 11, 11), cv::Size( -1, -1 ), criteria );
    }
    imagePoints.push_back( corners );

    //draw and show the corners
    cv::drawChessboardCorners( total_imgs[i], patternSize, corners, patternfound );
    cv::resize( total_imgs[i], img_resized, cv::Size(), 0.5, 0.5 );// just for better visualization..
    cv::namedWindow( PATH_NAME );
    cv::imshow( PATH_NAME, img_resized );
    cv::waitKey(20);
    cv::destroyWindow(PATH_NAME);

    // Fill objectPoints and imagePoints container.
    objectPoints.push_back(objPoints);
    imageSize = total_imgs[i].size();
    std::cout << imageSize << std::endl;
    std::cout << "DONE\n" << std::endl;

    // Image identification index increment.
    n++;
  }

  std::cout << "CALIBRATION SETTINGS DONE\n" << std::endl;
  std::cout << "-----------------------------------------------\n" << std::endl;
  std::cout << "START CALIBRATION \n" << std::endl;
  double rms = calibrateCamera( objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0, criteria_calib );
  std::cout << "CALIBRATION DONE\n" << std::endl;
  std::cout << "RMS REPROJECTION ERROR COMPUTED BY OPENCV CALIBRATECAMERA FUNCTION: " <<rms << "\n" << std::endl;
  std::cout << "-----------------------------------------------\n" << std::endl;


  // Reprojection error
  std::vector <double> errorContainer(IMG_NUM);
  double sumError;
  double normCoords;
  double total;
  int maxIndex = 0;
  int minIndex = 0;
  n = 1;

  std::cout << "REPROJECTION ERROR OF EACH IMAGE\n" <<  std::endl;
  for ( int i = 0; i<IMG_NUM; i++ )
  {
    cv::projectPoints( objPoints, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, prjPoints, cv::noArray(), 0 );
    corners = imagePoints[i];
    sumError = 0;

    for (int m=0; m<corners.size(); m++)
    {
      normCoords = 0;

      for ( int l=0; l<2; l++ )
      {
        double distCoords = pow( abs( corners[m][l] - prjPoints[m][l] ), 2 );
        normCoords = normCoords + distCoords;
      }

      sumError = sumError + normCoords;
    }

    errorContainer[i] = sumError/corners.size();
    total = total + errorContainer[i];
    errorContainer[i] = errorContainer[i];
    std::cout << "IMG_" << i+1 << ".png --> " << errorContainer[i] << "\n" <<  std::endl;

    // Computing calibration performance for each image.
    if ( errorContainer[i] > errorContainer[maxIndex] )
    {
      maxIndex = i;
    }
    if ( errorContainer[i] < errorContainer[minIndex] )
    {
      minIndex = i;
    }
    n++;

  }

  double mrpjError = total/IMG_NUM;
  std::cout << "MEAN (SQUARE) REPROJECTION ERROR OF CALIBRATION: " << mrpjError << "\n" <<  std::endl;
  std::cout << "ROOT MEAN (SQUARE) REPROJECTION ERROR OF CALIBRATION: " << sqrt(mrpjError) << "\n" << std::endl;
  std::cout << "-----------------------------------------------\n" << std::endl;

  // Worst and best images.
  sprintf ( PATH_NAME, "../checkerboard_test/IMG_%d.png", maxIndex+1 );
  std::cout << "WORST IMAGE: " << PATH_NAME << "\n" <<  std::endl;
  sprintf ( PATH_NAME, "../checkerboard_test/IMG_%d.png", minIndex+1 );
  std::cout << "BEST IMAGE: " << PATH_NAME << "\n" <<  std::endl;

  //Undistrorsion process.
  cv::Mat img = cv::imread("../test_image.png");
  cv::Mat imageUndistorted;
  cv::Mat newCameraMatrix;
  cv::Mat map1, map2;
  cv::Mat R;
  cv::initUndistortRectifyMap( cameraMatrix, distCoeffs, R, newCameraMatrix, imageSize, CV_32FC1, map1, map2 );
  cv::remap( img, imageUndistorted, map1, map2, cv::INTER_LINEAR );
  cv::resize( imageUndistorted, imageUndistorted, cv::Size(), 0.36, 0.36 );
  cv::resize( img, img, cv::Size(), 0.36, 0.36 );
  cv::namedWindow( "Before processing", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Before processing", img );
  cv::waitKey(0);
  cv::namedWindow( "After processing", CV_WINDOW_AUTOSIZE );
  cv::imshow( "After processing", imageUndistorted );
  cv::waitKey(0);

  // Alternatively we can use a split view...
  // cv::Mat concat;
  // cv::hconcat( img, imageUndistorted, concat );
  // cv::namedWindow( "Comparison", CV_WINDOW_AUTOSIZE );
  // cv::imshow( "Comparison", concat );
  // cv::waitKey(0);

}
