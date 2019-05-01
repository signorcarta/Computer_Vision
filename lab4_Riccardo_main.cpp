#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"

//parameters to detect lines
//canny parameters
#define MIN_TH 340
#define MAX_TH 850
#define KERNEL_CANNY 3
//HoughLines parameters
#define RHO 1
#define THETA (CV_PI/180)
#define INTERS_TH 130
//length of line
#define LENGTH_LINE 1000

//parameters to detect circles
//medianBlur parameter
#define KERNEL_MEDBLUR 3
//HoughCircles parameters
#define MIN_DIST 1
#define UPPER_TH 100
#define CENTER_TH 25
#define MIN_RADIUS 0
#define MAX_RADIUS 10

cv::Mat detection(void*userdata)
{
	cv::Mat image, img, gray, edge, draw, gray_circle;
	cv::Size imgSize;
	std::vector<cv::Vec2f> lines;
	std::vector<cv::Point> points, point;
	std::vector<cv::Vec3f> circles;

	image = *(cv::Mat*) userdata;
	img = image.clone();
	imgSize = img.size();
	
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	//detect lines
	cv::Canny(gray, edge, MIN_TH, MAX_TH, KERNEL_CANNY);
	cv::HoughLines(edge, lines, RHO, THETA, INTERS_TH);
	//detect circles
	cv::medianBlur(gray, gray_circle, KERNEL_MEDBLUR);
	cv::HoughCircles(gray_circle, circles, cv::HOUGH_GRADIENT, 1, MIN_DIST, UPPER_TH, CENTER_TH, MIN_RADIUS, MAX_RADIUS);

	//draw lines
	for (int i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + LENGTH_LINE * (-b));
		pt1.y = cvRound(y0 + LENGTH_LINE * (a));
		pt2.x = cvRound(x0 - LENGTH_LINE * (-b));
		pt2.y = cvRound(y0 - LENGTH_LINE * (a));
		//cv::line(img, pt1, pt2, cv::Scalar(0, 0, 255),1 , 16);
		points.push_back(pt1);
		points.push_back(pt2);
	}
	//calculate m1, m2, q1, q2 of two lines
	float m1, m2, q1, q2;
	int xc, yc, yf, xf1, xf2;
	m1 = ((float)points[0].y - (float)points[1].y) / ((float)points[0].x - (float)points[1].x);
	m2 = ((float)points[2].y - (float)points[3].y) / ((float)points[2].x - (float)points[3].x);
	q1 = ((float)((points[0].x) * points[1].y) - (float)(points[1].x * points[0].y)) / ((float)points[0].x - (float)points[1].x);
	q2 = ((float)(points[2].x * points[3].y) - (float)(points[3].x * points[2].y)) / ((float)points[2].x - (float)points[3].x);
	//calculating intersection points
	//intersection between two line
	xc = ((q1 - q2) / (m2 - m1));
	yc = ((m1 * xc) + q1);
	//intersection with line y=374 (max y value for this image)
	yf = imgSize.height - 1; // in this case yf=374
	xf1 = ((yf - q1) / m1);
	xf2 = ((yf - q2) / m2);

	point.push_back(cv::Point(xc, yc));  //point1
	point.push_back(cv::Point(xf1, yf));  //point2
	point.push_back(cv::Point(xf2, yf));  //point3
	cv::fillConvexPoly(img, point, cv::Scalar(0, 0, 255), 8, 0);

	
	//draw circles
	for (int i = 0; i < circles.size(); i++)
	{
		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		cv::circle(img, center, radius, cv::Scalar(0, 255, 0), -1, 8, 0);
	}

	return img;
}

int main(int argc, char*argv[])
{
	cv::Mat img1 = cv::imread("images/road2.png");
	if (img1.data == NULL)
	{
		std::cout << "Can't read the image" << std::endl;
		return 1;
	}
	cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original image", img1);
	
	cv::Mat img2 = detection((void*)&img1);

	cv::namedWindow("Output image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Output image", img2);

	cv::waitKey(0);
	return 0;
}
