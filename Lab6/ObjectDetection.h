#ifndef LAB6__OBJECT__DETECTION__H
#define LAB6__OBJECT__DETECTION__H
#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/core.hpp>

class ObjectDetection
{
   public:

      // Object istances
      cv::Mat imageTrain;
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;

      ObjectDetection( std::string path, std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors );

      static void SiftFeaturesExtractor (  ObjectDetection& obj );

      static void matcher( ObjectDetection& train, ObjectDetection& scene, std::vector<cv::DMatch>& matches );

      static void matchesRefiner( std::vector<cv::DMatch>& matches,
                                  std::vector<cv::DMatch>& goodMatches,
                                  double ratio );

      static void localizeInImage( ObjectDetection& train,
                                   ObjectDetection& scene,
                                   cv::Mat& img_matches,
                                   const std::vector<cv::DMatch>& goodMatches );

      static void inliersRetriver( ObjectDetection& train,
                                   ObjectDetection& scene,
                                   const std::vector<cv::DMatch>& goodMatches,
                                   std::vector<cv::DMatch>& inliersGoodMatches );
   private:



};

#endif
