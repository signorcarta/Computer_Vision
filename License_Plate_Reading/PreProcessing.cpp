#include "PreProcessing.h"

using namespace cv;
using namespace std;

void Preprocess(Mat& srcImage, Mat& processed) {

	Mat grayImage;
	Mat ImageMaxContrast;
	Mat imgFiltered;
	Mat thresholdImage;
	Mat structuringElement = getStructuringElement(MORPH_RECT, Size(2, 2));

	/// Get the grayscale image
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	/// Apply maxContrast() function
	ImageMaxContrast = maxContrast(grayImage); 
	/// Perform a gaussian blur
	GaussianBlur(ImageMaxContrast, imgFiltered, Size(5, 5), 0);																						
	/// Transform grayImage to a binary image
	threshold(imgFiltered, thresholdImage, 0, 255, THRESH_BINARY_INV+THRESH_OTSU); 
	/// Perform dilation of the image
	morphologyEx(thresholdImage, processed, MORPH_DILATE, structuringElement, Point(-1,1), 2);
	
}

Mat maxContrast(Mat grayImage) {
	/*

	CLOSING: is the reverse of Opening, Dilation followed by Erosion.
	It is used to close small holes inside the foreground objects,
	or small black points on the object.

	OPENING: it's erosion followed by dilation.

	*/

	Mat imgTopHat; /// difference between INPUT image and OPENING of the image
	Mat imgBlackHat; /// difference between the CLOSING of the input image and INPUT image
	Mat structuringElement = getStructuringElement(MORPH_RECT, Size(3, 3)); /// Structuring element for morphological operations.
	morphologyEx(grayImage, imgTopHat, MORPH_TOPHAT, structuringElement);
	morphologyEx(grayImage, imgBlackHat, MORPH_BLACKHAT, structuringElement);

	Mat grayImagePlusTopHatMinusBlackHat;
	grayImagePlusTopHatMinusBlackHat = grayImage + imgTopHat - imgBlackHat;

	return(grayImagePlusTopHatMinusBlackHat);
}







