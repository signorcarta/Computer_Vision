#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

#include "PreProcessing.h"
#include "PlateDetection.h"
#include "PlatesExtraction.h"
//#include "classifier/haarcascade_russian_plate_number.xml"



#define SHOW_STEPS /// Comment/uncomment this line to hide/show intermediate steps 
//#define DISABLE /// If preprocess is disabled ==> change input image in detectPlate()

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	//Loading image________________________________________________________________________________

	Mat src = imread("C:\\Users\\david\\source\\repos\\License_plate_recognition\\5.jpg"); /// Source image
	
#ifdef SHOW_STEPS
	cout << "---> Showing original image. \n\n" << endl;
	namedWindow("ORIGINAL IMAGE");
	imshow("ORIGINAL IMAGE", src);
	waitKey();
#endif //SHOW_STEPS

	//_____________________________________________________________________________________________
	
	/*
	
		//Preprocessing of src image___________________________________________________________________

	#ifdef PROCESS
		Mat thresholded; ///Thresholded image

		Preprocess(src, thresholded);

	#ifdef SHOW_STEPS
		cout << "---> Showing thresholded image. \n\n" << endl;
		namedWindow("THRESHOLDED IMAGE");
		imshow("THRESHOLDED IMAGE", thresholded);
		waitKey();
	#endif
	#endif

		//_____________________________________________________________________________________________

	*/
	


	//Plate detection using cascade classifier_____________________________________________________

	Mat detected; /// Image with the drawn rectangles
	vector<Rect> rects; /// Vector of rectangles representing detected plates
	int n_plates = 0; /// Number of detected plates
	String path = "C:\\Users\\david\\source\\repos\\License_plate_recognition\\classifier\\haarcascade_russian_plate_number.xml";
	
	detectPlate(src, detected, path, n_plates, rects);

#ifdef SHOW_STEPS
	cout << "\n---> Showing possible detected plates.\n";
	if (n_plates<2) {cout << "     Found just " << n_plates << " plate.\n\n" << endl;}
	else {cout << "     Found " << n_plates << " possible plates.\n" << endl;}	
	namedWindow("DETECTED PLATE");
	imshow("DETECTED PLATE", detected);
	waitKey();
#endif //SHOW_STEPS

	//_____________________________________________________________________________________________




	//Plate extraction_____________________________________________________________________________

	vector<Mat> vecOfPlates; /// Vector containing possible plates cropped from the image

	extractPlate(detected, rects, vecOfPlates, n_plates);

#ifdef SHOW_STEPS
	showPlate(vecOfPlates);
#endif //SHOW_STEPS

	//_____________________________________________________________________________________________
	

	//Plate thresholding___________________________________________________________________________
	vector<Mat> thresholded_plate(n_plates); ///Thresholded plate

	for (int i = 0; i<n_plates; i++) {
		Preprocess(vecOfPlates[i], thresholded_plate[i]);

	#ifdef SHOW_STEPS
		cout << "---> Showing thresholded plate. \n\n" << endl;
		namedWindow("THRESHOLDED PLATE");
		imshow("THRESHOLDED PLATE", thresholded_plate[i]);
		waitKey();
	#endif //SHOW_STEPS
	}
	
	//_____________________________________________________________________________________________

	///Get horizontal and vertical histograms______________________________________________________

	vector<Mat> vertical(n_plates); /// Vertical histograms  of each detected plate
	vector<Mat> horizontal(n_plates); /// Horizontal histograms of each detected plate
	
		for (int i = 0; i < n_plates; i++) {			

			getHistograms(thresholded_plate[i], vertical[i], horizontal[i]);	
			
		}

		
		/*
			/////////////////////////////////////////////////////////////////////////////////////////////////
			//                                                                                             //
			// Need to figure out what the fuck is going on in (*) and (**) and why I can't get the Mat(s) //
			// in verticale and orizzontale respectively                                                   //
			//                                                                                             //
			/////////////////////////////////////////////////////////////////////////////////////////////////
		*/


		for (int i = 0; i < n_plates; i++) {

			Mat verticale = vertical[i]; // (*)
			Mat orizzontale = horizontal[i]; // (**)

		
			cout << "verticale size: " << verticale.size() << endl;
			cout << "orizzontale size: " << orizzontale.size() << endl;
		

#ifdef DISABLE
			cout << "---> Showing VERTICAL histogram. \n\n" << endl;
			namedWindow("VERTICAL HISTOGRAM");
			imshow("VERTICAL HISTOGRAM", verticale);
			waitKey();

			cout << "---> Showing HORIZONTAL histogram. \n\n" << endl;
			namedWindow("HORIZONTAL HISTOGRAM");
			imshow("HORIZONTAL HISTOGRAM", horizontal);
			waitKey();
#endif // DISABLE
		}

		///____________________________________________________________________________________________



	
	return 0;
}
