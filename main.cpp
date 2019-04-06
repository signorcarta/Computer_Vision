#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define n_imgs 15	
#define SQ_SIZE 0.004 //meters [to be verifed]
#define INNER_ROWS 8
#define INNER_COLS 12

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	
	//inizialization of input images vector. Memory allocation.
	cv::Mat src[n_imgs];
	
	// (1).Loading the images______________________________________________________________________________________________
	
	vector<string> fn; 
	string path = format("../Huawei-Pietro/img*.jpg"); 
	glob(path, fn, false);  
	for (size_t i = 0; i < fn.size(); ++i){		

		//read i-th image
		Mat src = imread(fn[i]); 
		
		/*_______________________________________________________________________________________
		namedWindow("Display window number " + i, WINDOW_AUTOSIZE); //Create a window for display.
		imshow("Display window", src); //Show our image inside it.
		waitKey(30);_____________________________________________________________________________
		*/

		// Size of the checkerboard's inner pattern
		cv::Size patternSize = cv::Size(INNER_ROWS, INNER_COLS); 		
		// Vector of detected corners coordinates of an image. This will be filled by the detected corners
		std::vector<cv::Vec2f> corners; 

		// (2).Pattern detection___________________________________________________________________________________________
		
		bool patternfound = findChessboardCorners(src, patternSize, corners);
		//Source chessboard image initialization.
		cv::Mat grayImage;
		//Termination criteria for the iterative optimization algorithm.
		cv::TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.00001);

		if (patternfound) {
						
			cv::cvtColor(src, grayImage, COLOR_BGR2GRAY); //Source chessboard image to grayscale image.
			cv::cornerSubPix(grayImage, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria); //function to refine the corner detections
			drawChessboardCorners(grayImage, patternSize, Mat(corners), patternfound);

		}

		// Vector of detected corners of all images.
		std::vector< std::vector<cv::Vec2f> > imagePoints;
		//fill the fucking vector(???)//		
	}
}



