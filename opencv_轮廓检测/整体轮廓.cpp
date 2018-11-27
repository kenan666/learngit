
//测试时候用的集合图形，完全符合要求
#include<cv.h>  
#include<highgui.h>  

using namespace cv;
using namespace std;

int main()
{
	IplImage* src = NULL;
	IplImage* img = NULL;
	IplImage* dst = NULL;

	CvMemStorage* storage = cvCreateMemStorage(0);;
	CvSeq* contour = 0;
	int contours = 0;
	CvScalar external_color;
	CvScalar hole_color;

	src = cvLoadImage("t1-1.png", 1);
	img = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);

	cvCvtColor(src, img, CV_BGR2GRAY);
	cvThreshold(img, img, 100, 200, CV_THRESH_BINARY);
	contours = cvFindContours(img, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	for (; contour != 0; contour = contour->h_next)
	{
		external_color = CV_RGB(rand() & 255, rand() & 255, rand() & 255);
		hole_color = CV_RGB(rand() & 255, rand() & 255, rand() & 255);
		cvDrawContours(dst, contour, external_color, hole_color, 1, 2, 8);
	}

	cvNamedWindow("Contour", 1);
	cvShowImage("Contour", dst);

	cvWaitKey(0);

	cvReleaseMemStorage(&storage);
	cvReleaseImage(&src);
	cvReleaseImage(&img);
	cvReleaseImage(&dst);

	return 0;
}
