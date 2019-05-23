#ifndef LAB5__PANORAMIC__UTILS__H
#define LAB5__PANORAMIC__UTILS__H

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class PanoramicUtils{

public:

	static Mat cylindricalProj(const Mat& image, const double angle);
};

#endif
