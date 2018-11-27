
#include <opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

void main() 
{
	//图片保存参数
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(9);

	cv::Mat src = cv::imread("G:/opencv 3.2 file/OPENCV FILE/OPENCV FILE/t1-1.png", 1);

	Mat binary = Mat::zeros(src.size(), CV_8UC1);

	//色彩分割得到二值图像

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

	//对二值图像进行膨胀操作
	Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	Mat dilate;
	cv::dilate(binary, dilate, element);
	imshow("dilate", dilate);
	cv::imwrite("dilate.jpg", dilate, compression_params);

	//膨胀之后的图像减去膨胀前的二值图像就是物体的轮廓
	Mat edge = dilate - binary;
	imshow("edge", edge);
	cv::imwrite("edge.jpg", edge, compression_params);

	cv::waitKey(0);
}
