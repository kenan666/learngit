#include<iostream>
#include<opencv2\opencv.hpp>  
//#include<opencv2\nonfree\nonfree.hpp>  
using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
	//ͼ���ȡ  
	Mat img1, img2;
	img1 = imread("s1_1_1.png", CV_WINDOW_AUTOSIZE);
	img2 = imread("s4_1.png", CV_WINDOW_AUTOSIZE);

	//sift������ȡ  
	SiftFeatureDetector detector;
	vector<KeyPoint> keyPoint1, keyPoint2;
	detector.detect(img1, keyPoint1);
	detector.detect(img2, keyPoint2);
	cout << "Number of KeyPoint1:" << keyPoint1.size() << endl;
	cout << "Number of KeyPoint2:" << keyPoint2.size() << endl;

	//sift���������Ӽ���  
	SiftDescriptorExtractor desExtractor;
	Mat des1, des2;
	desExtractor.compute(img1, keyPoint1, des1);
	desExtractor.compute(img2, keyPoint2, des2);

	//sift������(������)ƥ��  
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(des1, des2, matches);

	Mat img_match;
	drawMatches(img1, keyPoint1, img2, keyPoint2, matches, img_match);

	imshow("img_match", img_match);

	waitKey(0);

	return 0;
}

