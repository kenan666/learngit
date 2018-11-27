
#include <opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

void main() 
{
	//ͼƬ�������
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(9);

	cv::Mat src = cv::imread("G:/opencv 3.2 file/OPENCV FILE/OPENCV FILE/t1-1.png", 1);

	Mat binary = Mat::zeros(src.size(), CV_8UC1);

	//ɫ�ʷָ�õ���ֵͼ��

	for (int ii = 0; ii<src.rows; ii++)

		for (int jj = 0; jj<src.cols; jj++) 
		{
			int b = (int)(src.at<Vec3b>(ii, jj)[0]);
			int r = (int)(src.at<Vec3b>(ii, jj)[2]);

			if (r>150 & b<100) 
			{
				binary.at<uchar>(ii, jj) = 255;
			}
			else binary.at<uchar>(ii, jj) = 0;
		}

	cv::imshow("binary", binary);
	cv::imwrite("binary.jpg", binary, compression_params);

	//�Զ�ֵͼ��������Ͳ���
	Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	Mat dilate;
	cv::dilate(binary, dilate, element);
	imshow("dilate", dilate);
	cv::imwrite("dilate.jpg", dilate, compression_params);

	//����֮���ͼ���ȥ����ǰ�Ķ�ֵͼ��������������
	Mat edge = dilate - binary;
	imshow("edge", edge);
	cv::imwrite("edge.jpg", edge, compression_params);

	cv::waitKey(0);
}
