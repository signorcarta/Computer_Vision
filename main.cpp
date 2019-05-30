#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>



using namespace cv;

int main(int argc, char** argv) {
	Mat src = imread("C:\\Users\\david\\source\\repos\\Licence_plate_reading\\image1.png");
	namedWindow("Originale image");
	imshow("Originale image", src);
	waitKey();

	return 0;
}