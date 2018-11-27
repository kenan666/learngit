
//#include "stdafx.h"  
//��ʱ�����ܹ�ʹ��
#include <opencv2/opencv.hpp>  
#include <iostream>  
#include <stack>  

#include <cv.h>  
#include <highgui.h>  

using namespace std;
using namespace cv;

//�����������  
int GetLayerCnt(CvSeq *seq)
{
	int count = 0;
	while (seq->v_next)
	{
		count++;
		seq = seq->v_next;
	}
	return count;
}

//��ȡֻ��һ������ͨ������  
void GetAllContourByLimit(CvSeq *contour, vector<CvSeq *> &vecSeq)
{
	int count = 0, count2 = 0;
	CvSeq *contourNoLevel = 0, *contourSameLevel = 0;
	stack<CvSeq *> data;
	CvSeq *node = 0, *tmpContour = 0;
	data.push(contour);
	while (!data.empty())
	{
		node = data.top();
		data.pop();
		tmpContour = node->h_next;
		//brother  
		if (tmpContour)
		{
			data.push(tmpContour);
		}
		//child  
		count = GetLayerCnt(node);
		if (count >= 2)// ��Ҫ��ȡ ���Ϊn ������������ count >= n+1 .  
		{
			contourNoLevel = node->v_next->v_next;// �˴�����count��ֵ  
			if (!contourNoLevel->v_next && !contourNoLevel->h_next)
			{
				vecSeq.push_back(node);
			}
			if (contourNoLevel->h_next || contourNoLevel->v_next)
			{
				data.push(contourNoLevel);
			}
		}
		else if (count == 1)
		{
			tmpContour = node->v_next->h_next;
			if (tmpContour)
			{
				data.push(tmpContour);
			}
		}
	}
}

//ץȡ��������  
void GetContourCenter(CvSeq *contour, CvPoint &p)
{
	//���ķ�ץ���ĵ�  
	int contourlength = contour->total;
	CvPoint *pt = 0;
	double avg_px = 0, avg_py = 0;
	for (int i = 0; i < contourlength; i++)
	{
		pt = CV_GET_SEQ_ELEM(CvPoint, contour, i);
		avg_px += pt->x;
		avg_py += pt->y;
	}
	p.x = avg_px / contourlength;
	p.y = avg_py / contourlength;
}

//������  
int main(int argc,char ** argv[])
{
	IplImage * src;
	// the first command line parameter must be file name of binary (black-n-white) image  
	if ((src = cvLoadImage("G:\\opencv 3.2 file\\OPENCV FILE\\OPENCV FILE\\ceshi.png", 0)) != NULL)
	{
		IplImage * dst = cvCreateImage(cvGetSize(src), 8, 3);
		CvMemStorage * storage = cvCreateMemStorage(0);
		CvSeq* contour = 0, *contourNoLevel = 0, *contourSameLevel = 0, *outpput = 0;

		cvThreshold(src, src, 1, 255, CV_THRESH_OTSU);
		cvNot(src, src);
		cvFindContours(src, storage, &contour, sizeof(CvContour), 3, CV_CHAIN_APPROX_NONE);
		cvZero(dst);
		vector<CvSeq *> vecSeq;
		vecSeq.clear();
		//��ȡ����ֻ��һ������ͨ������  
		GetAllContourByLimit(contour, vecSeq);
		//��ʾ  
		CvPoint pt;
		for (int i = 0; i<vecSeq.size(); i++)
		{
			if (1)//�˴����������������  
			{
				GetContourCenter(vecSeq[i]->v_next->v_next, pt);
				cvCircle(dst, pt, 2, cvScalar(0, 255, 0), 1, 8, 0);
				cvDrawContours(dst, vecSeq[i], CV_RGB(255, 0, 0), CV_RGB(0, 0, 0), -1, 2);
			}
		}
		cvNamedWindow("Result");
		cvShowImage("Result", dst);
		cvWaitKey(0);
	}
	system("pause");
	return 0;
}
