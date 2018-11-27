

//寻找是圆轮廓的图形。。。。

#include <iostream>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 

using namespace cv;
using namespace std;

int main()
{
	Mat q_MatImage;
	Mat q_MatImageGray;
	Mat q_MatImageShow;
	Mat q_MatImageShow2;
	q_MatImage = imread("ceshi.png");//读入一张图片
	q_MatImage.copyTo(q_MatImageShow);
	q_MatImage.copyTo(q_MatImageShow2);
	cvtColor(q_MatImage, q_MatImageGray, CV_RGB2GRAY);
	double q_dEpsilon = 10E-9;
	unsigned int q_iReturn = 0;

	int q_iX, q_iY, q_iWidth, q_iHeight;
	q_iX = 20;
	q_iY = 40;
	q_iWidth = 600;
	q_iHeight = 420;

	double q_dThresholdSimilarity = 60;
	double q_dThresholdMin = 35;
	double q_dThresholdMax = 75;

	threshold(q_MatImageGray, q_MatImageGray, 150, 255, CV_THRESH_BINARY);

	namedWindow("Test1");		//创建一个名为Test窗口
	imshow("Test1", q_MatImageGray);			//窗口中显示图像

	vector<vector<Point>> q_vPointContours;

	findContours(q_MatImageGray, q_vPointContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

	size_t q_iAmountContours = q_vPointContours.size();

	for (size_t i = 0; i < q_iAmountContours; i++)
	{
		size_t q_perNum = q_vPointContours[i].size();
		for (size_t j = 0; j < q_iAmountContours; j++)
		{
			circle(q_MatImageGray, q_vPointContours[i][j], 3, CV_RGB(0, 255, 0), 1, 8, 3);
		}
	}

	namedWindow("findContours");
	imshow("findContours", q_MatImageGray);

	std::vector<cv::Point2f> q_vPointCentersContours(q_iAmountContours);
	std::vector<double> q_vdRadiusesContours(q_iAmountContours);
	std::vector<double> q_vdSimilarityContours(q_iAmountContours);
	std::vector<bool> q_vbFlagCircles(q_iAmountContours);

	std::vector<double> q_vdRadiusesContour;
	double q_dRadiusMax, q_dRadiusMin;
	double q_dSumX, q_dSumY;
	size_t q_iAmountPoints;

	for (size_t q_iCycleContours = 0; q_iCycleContours<q_iAmountContours; q_iCycleContours++)
	{
		q_dSumX = 0.0;
		q_dSumY = 0.0;
		q_iAmountPoints = q_vPointContours[q_iCycleContours].size();
		if (0 >= q_iAmountPoints)
		{
			continue;
		}
		for (size_t q_iCyclePoints = 0; q_iCyclePoints<q_iAmountPoints; q_iCyclePoints++)
		{
			q_dSumX += q_vPointContours[q_iCycleContours].at(q_iCyclePoints).x;
			q_dSumY += q_vPointContours[q_iCycleContours].at(q_iCyclePoints).y;
		}

		q_vPointCentersContours[q_iCycleContours].x = (float)(q_dSumX / q_iAmountPoints);//均值中心点X</span>
		q_vPointCentersContours[q_iCycleContours].y = (float)(q_dSumY / q_iAmountPoints);//均值中心点Y</span>


		q_vdRadiusesContour.resize(q_iAmountPoints);
		double q_dDifferenceX, q_dDifferenceY;
		double q_dSumRadius = 0.0;
		q_dRadiusMax = 0.0;
		q_dRadiusMin = DBL_MAX;;
		for (size_t q_iCyclePoints = 0; q_iCyclePoints<q_iAmountPoints; q_iCyclePoints++)
		{
			q_dDifferenceX = q_vPointCentersContours[q_iCycleContours].x - q_vPointContours[q_iCycleContours].at(q_iCyclePoints).x;
			q_dDifferenceY = q_vPointCentersContours[q_iCycleContours].y - q_vPointContours[q_iCycleContours].at(q_iCyclePoints).y;
			q_vdRadiusesContour[q_iCyclePoints] = sqrt(q_dDifferenceX*q_dDifferenceX + q_dDifferenceY*q_dDifferenceY);

			if (q_vdRadiusesContour[q_iCyclePoints]>q_dRadiusMax)
			{
				q_dRadiusMax = q_vdRadiusesContour[q_iCyclePoints];
			}
			if (q_vdRadiusesContour[q_iCyclePoints]<q_dRadiusMin)
			{
				q_dRadiusMin = q_vdRadiusesContour[q_iCyclePoints];
			}

			q_dSumRadius += q_vdRadiusesContour[q_iCyclePoints];
		}
		q_vdRadiusesContours[q_iCycleContours] = q_dSumRadius / q_iAmountPoints;   //均值半径

		q_vdSimilarityContours[q_iCycleContours] = 100.0*q_dRadiusMin / q_dRadiusMax;  //相似度
		if ((q_dThresholdSimilarity<q_vdSimilarityContours[q_iCycleContours]) &&
			(q_dThresholdMin<q_vdRadiusesContours[q_iCycleContours]) &&
			(q_dThresholdMax>q_vdRadiusesContours[q_iCycleContours]))    //判断是否是圆
		{
			q_vbFlagCircles[q_iCycleContours] = true;
		}
		else
		{
			q_vbFlagCircles[q_iCycleContours] = false;
		}
	}


	if (q_dEpsilon < 10)
	{
		cv::Point q_PointCenterCurrent;
		for (size_t q_iCycleContours = 0; q_iCycleContours<q_iAmountContours; q_iCycleContours++)
		{
			if (q_vbFlagCircles[q_iCycleContours])
			{
				q_PointCenterCurrent.x = (int)(q_vPointCentersContours[q_iCycleContours].x);
				q_PointCenterCurrent.y = (int)(q_vPointCentersContours[q_iCycleContours].y);
				circle(q_MatImageShow, q_PointCenterCurrent, 3, Scalar(0.0, 0.0, 255.0), 0);
			}
		}
	}

	int q_iIndexResultBegin = 4;
	int q_iAmountCircleResult = 4;
	int q_iIndexCiecleCurrent;
	int q_iCountCircles = 0;

	for (size_t q_iCycleContours = 0; q_iCycleContours<q_iAmountContours; q_iCycleContours++)
	{
		if (q_vbFlagCircles[q_iCycleContours])
		{
			q_iIndexCiecleCurrent = q_iIndexResultBegin + q_iAmountCircleResult*q_iCountCircles;
			q_iCountCircles++;
		}
	}
	cout << "总共找到 " << q_iCountCircles << "个圆！" << endl;


	namedWindow("Test");		//创建一个名为Test窗口
	imshow("Test", q_MatImageShow);//窗口中显示图像
	waitKey();				//等待5000ms后窗口自动关闭
}
