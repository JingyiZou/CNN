#include<iostream>
#include"CnnForwardPass.h"
#include"otherfunction.h"
#include"readImage.h"
#include<ctime>
#include <omp.h>
using namespace std;

int main()
{
	Mat src = imread("test1.jpg");
	CMyCNN cnn;
	cnn.imagedata = Mat2pChar(src);
	cnn.inputwidth = src.cols;
	cnn.inputheight = src.rows;
	cnn.inputband = src.channels();
	cnn.isOpenMp = true;
	auto t1 = clock();
	//double num = 0;
	//for (int i = 0; i < 100;i++)
	//{
	auto labelres = cnn.ForwardPass(cnn.imagedata);
		//cout << "ִ�н��: " << labelres << endl;
		//if (labelres)
			//num++;
	//}
	cout <<"����ִ��ʱ��s��" << (clock()-t1)*1.0/CLK_TCK << endl;	//�õ����������ִ��ʱ�䣬��λ����
	//cout << "��ȷ��=" << num / 100 << endl;
	cout << "ִ�н��: " << labelres << endl;

//#pragma omp parallel
//	{
//		cout << "Hello" << ", I am Thread " << omp_get_thread_num() << endl;
//	}

	system("pause");

	return 1;
}


