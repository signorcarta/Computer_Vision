#include "charsExtraction.h"


using namespace std;
using namespace cv;


void extractChars(Mat& image, vector<Rect>& charsRects, vector<Mat>& charsCollection_resized) {

	if (!charsRects.empty()) {

		Mat image_ = image.clone();
		vector<Rect> charsRects_;
		vector<Mat> charsCol;


		///Get a copy of the array of rects
		for (int i = 0; i < charsRects.size(); i++) {

			charsRects_.push_back(charsRects[i]);
		}

		/// Sort the array of rectangles__________________________

		struct byPosition {

			bool operator() (Rect& a, const Rect& b) {
				return a.x < b.x;
			}

		};

		sort(charsRects_.begin(), charsRects_.end(), byPosition());

		///_______________________________________________________

		/// FIll the output vector with ordered chars_____________

		for (int i = 0; i < charsRects_.size(); i++) {

			Mat original = image_.clone();
			charsCol.push_back(original(charsRects_[i]));

		}
		///_______________________________________________________


		///Resize to 24x24 and perform a binary thresholding______

		for (int i = 0; i < charsCol.size(); i++) {

			Mat res = charsCol[i].clone();
			cvtColor(res, res, COLOR_BGR2GRAY);
			threshold(res, res, 220, 255, THRESH_BINARY);
			resize(res, res, Size(24, 24));
			charsCollection_resized.push_back(res);
		}

		///_______________________________________________________
	}


}


void showChars(vector<Mat>& charsCollection_resized) {

	if (!charsCollection_resized.empty()) {

		for (int i = 0; i < charsCollection_resized.size(); i++) {

			Mat temp = charsCollection_resized[i].clone();

			namedWindow("Char");
			imshow("Char", temp);
			waitKey();
		}

	}

}

