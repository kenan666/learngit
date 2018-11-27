
//�߽����Ӳ�������ȫ�����ҳ��������ܹ��ҳ����±߽�

#include <cv.h> 
#include <cxcore.h> 
#include <highgui.h> 
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <iostream>  

using namespace std;
using namespace cv;

int main(int argc, char ** argv)
{
	IplImage * pImg = NULL;
	IplImage * pCannyImg = NULL;

	cv::Mat src = cv::imread("t1-1.png");
	if (src.empty())
		return -1;

	cv::Mat bw;
	cv::cvtColor(src, bw, CV_BGR2GRAY);
	Mat canny_mat(src.size(), CV_8U);

	//cvCanny(pImg,pCannyImg,50,150,3); 
	cv::Canny(bw, canny_mat, 50, 150, 3);

	imshow("canny", canny_mat);

	cvWaitKey(0); //�ȴ�����
	return 0;

}
