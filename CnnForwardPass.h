#pragma once
#include<vector>
#include<iostream>
using namespace std;
/******************************************
        name:CMyCNN
		author:Jane
		brief:
		   简易卷积神经网络
******************************************/
class CMyCNN
{
 public:
	 CMyCNN()
	 {
		 ReadParm("layer1.parm",1);
		 ReadParm("layer2.parm",2);
		 ReadParm("layer3.parm",3);
		 ReadParm("layer4.parm",4);
		 ReadParm("layer5.parm",5);
	 }
	 ~CMyCNN()
	 {
		 delete[] Convparm1;
		 delete[] Convparm2;
		 delete[] Convparm3;
		 delete[] Convparm4;
		 delete[] Convparm5;

		 delete[] Bparm1;
		 delete[] Bparm2;
		 delete[] Bparm3;
		 delete[] Bparm4;
		 delete[] Bparm5;
	 }

	 //输入图像数据信息
	 int inputwidth;	//宽度
	 int inputheight;	//高度
	 int inputband;		//通道数
	 float **imagedata;	//图像数据指针

	 bool isOpenMp;		//是否开启OpenMP加速

	 int ForwardPass(float **imagedata);	//网络正向预测

 private:
	  //每一层数据结构体
     typedef struct LAYER
     {
	  float** data;	//输出值
	  int Width;    //矩阵的宽
	  int Height;   //矩阵的高
	  int D;        //矩阵的维度
     }layer;


	 //STL容器，理解成动态数组
	 vector<layer> m_CNNLayer;	//保存每一层的信息，总的层信息

	 bool Convolution(float **input2D,int inputwidth,int inputheight,int inputD,float*f,int size,int featurenum,int stride,float *d,int Relu,int Pool, int size_pool,int stride_pool);	//卷积操作、ReLU操作、池化操作
	 float** New2DMatrix_double(int nRow, int nColumn);		//开辟数组空间
	 void DelCNNLayer();	//释放所有层的空间
	 void DisplayLayer(layer l);	//显示当前层的信息
	 void ReadParm(string filename,int num);	//从txt中读取网络参数，读取参数初始化网络层参数

	 //卷积参数和偏置参数
	 float* Convparm1,* Convparm2,* Convparm3,* Convparm4,* Convparm5;
	 float* Bparm1,* Bparm2,* Bparm3,* Bparm4,* Bparm5;
};