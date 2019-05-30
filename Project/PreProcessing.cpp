#include "PreProcessing.h"

using namespace cv;

void Preprocess(Mat &srcImage, Mat &tresholdImage) {
	
	Mat grayImage;
	Mat ImageMaxContrast;
	Mat imgFiltered;

	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	ImageMaxContrast = maxContrast(grayImage); /// Apply maxContrast() function
	GaussianBlur(ImageMaxContrast, imgFiltered, Size(5, 5), 0);/// Perform a gaussian blur																						
	adaptiveThreshold(imgFiltered, tresholdImage, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 19, 9); /// Transform grayImage to a binary image
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
