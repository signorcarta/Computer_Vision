// lab1_a.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

#include "pch.h"
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

#define NEIGHBORHOOD_SIZE 9

/*
This function detects the left button click on the mouse and returns coordinates of the event
*/
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked at position (" << x << ", " << y << ")" << endl;
		
		/*
		Retrieving the image from the main.
		How to get the data from void* userdata?
		1) Need a cast from void* back to cv::Mat*
		2) Be careful, it is a void* not a void! Need to get the pointed data.

		//cv::Mat* imgpointer = static_cast<cv::Mat*>(userdata);
		//cv::Mat image = *imgpointer;
		*/
		cv::Mat image = *(cv::Mat*)userdata;

		std::cout << "\nimage dimension :" << image.cols << "x" << image.rows << std::endl;
		std::cout << "\nrow :" << x << std::endl;
		std::cout << "\ncolumn :" << y << std::endl;

		// need to clone to avoid overwriting input data
	        cv::Mat image_clone = image.clone();

		// Mean on the neighborhood
		int shift = NEIGHBORHOOD_SIZE / 2;
		int r1 = y - shift;
		int r2 = y + shift;
		int c1 = x - shift;
		int c2 = x + shift;
		if (r1 < 0) r1 = 0;
		if (r2 > image_clone.rows) r2 = image_clone.rows;
		if (c1 < 0) c1 = 0;
		if (c2 > image_clone.cols) c2 = image_clone.cols;

		/* 
		Here I select the squared chunk of the image that has area (NEIGHBORHOOD_SiZE)^2
		There is a +1 term because Range excludes the last index and r2 and c2 are respectively, 
		the last row and the last column we want to consider.
		*/
		image_clone = image_clone(cv::Range(r1, r2 + 1), cv::Range(c1, c2 + 1));

		//Initialization of the variables that will contain the mean of each foundamental color
		double meanB = 0;
		double meanG = 0;
		double meanR = 0;

		/*
		Computing the means.
		Summing over all pixels of the squared chunk and dividing by the area of it.
		*/
		for (int i = 0; i < image_clone.rows; i++)
		{
			for (int j = 0; j < image_clone.cols; j++)
			{
				meanB = meanB + image_clone.at<cv::Vec3b>(i, j)[0];
				meanG = meanG + image_clone.at<cv::Vec3b>(i, j)[1];
				meanR = meanR + image_clone.at<cv::Vec3b>(i, j)[2];
			}
		}

		meanB = round(meanB / (NEIGHBORHOOD_SIZE*NEIGHBORHOOD_SIZE));
		meanG = round(meanG / (NEIGHBORHOOD_SIZE*NEIGHBORHOOD_SIZE));
		meanR = round(meanR / (NEIGHBORHOOD_SIZE*NEIGHBORHOOD_SIZE));

		/*
		std::cout << meanB << std::endl;
		std::cout << meanG << std::endl;
		std::cout << meanR << std::endl;
		*/

		 /*
		 Color segmentation.
		 "The goal of segmentation is to simplify and/or change the 
		 representation of an image into something that is more 
		 meaningful and easier to analyze" 
		 Wikipedia
		 */
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				if (abs(meanB - image.at<cv::Vec3b>(i, j)[0]) < 50 &&
					abs(meanG - image.at<cv::Vec3b>(i, j)[1]) < 50 &&
					abs(meanR - image.at<cv::Vec3b>(i, j)[2]) < 50
					)
				{
					image.at<cv::Vec3b>(i, j)[0] = 92;
					image.at<cv::Vec3b>(i, j)[1] = 201;
					image.at<cv::Vec3b>(i, j)[2] = 37;
				}
			}
		}

		/*
		Final result
		*/
		cv::namedWindow("final_result", CV_WINDOW_AUTOSIZE);
		cv::imshow("final_result", image);
		cv::waitKey(0);

	}
	
	
}

int main(int argc, char** argv)
{
	/*
	Loading the image.

	This function automatically allocates the memory needed
	for the image data structure.
	*/	
	cv::Mat img = cv::imread("Robocup.jpg");

	/*
	Creating a window.

	This function opens a window on the screen that	contains and display an image.
	The first argument assigns a name to the window, the second one
	may be set either to 0 (default) or to CV_WINDOW_AUTOSIZE. In the former 
	case, the size of the window will be the same regardless of the image size,
	and the image will be scaled to fit within the window. In the latter case, 
	the window will expand or contract automatically when an image is loaded 
	according to the image’s true size.
	*/	
	cv::namedWindow("Robocup", CV_WINDOW_AUTOSIZE);

	/*
	Displaying the image.

	This function allows to display an image in a existing window.
	*/
	cv::imshow("Robocup", img);

	//set the callback function for mouse event
	cv::setMouseCallback("Robocup", CallBackFunc, (void*)&img);

	/*
	If a positive argument is given, the program will wait for that number
	of milliseconds and then continue even if nothing is pressed.
	If the argument is set to 0 or to a negative number, the program will
	wait indefinitely for a keypress.
	*/
	cv::waitKey(0);

	return 0;
}



//________________________________________________________________________________________________________________
// Per eseguire il programma: CTRL+F5 oppure Debug > Avvia senza eseguire debug
// Per eseguire il debug del programma: F5 oppure Debug > Avvia debug

// Suggerimenti per iniziare: 
//   1. Usare la finestra Esplora soluzioni per aggiungere/gestire i file
//   2. Usare la finestra Team Explorer per connettersi al controllo del codice sorgente
//   3. Usare la finestra di output per visualizzare l'output di compilazione e altri messaggi
//   4. Usare la finestra Elenco errori per visualizzare gli errori
//   5. Passare a Progetto > Aggiungi nuovo elemento per creare nuovi file di codice oppure a Progetto > Aggiungi elemento esistente per aggiungere file di codice esistenti al progetto
//   6. Per aprire di nuovo questo progetto in futuro, passare a File > Apri > Progetto e selezionare il file con estensione sln
