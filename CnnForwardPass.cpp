#include"stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include "CnnForwardPass.h"

// openmp加速头文件
#include <omp.h>

using namespace std;


/************************************************
	name：ForwardPass
	author：Jane
	brief：实现正向传播，输出预测结果与label一致
	param:
		imagedata：输入图像
************************************************/
int CMyCNN::ForwardPass(float **imagedata)
{
	int label;
	//block1
	Convolution(imagedata,61,61,3,Convparm1,5,4,1,Bparm1,1,1,3,2);

	//block2
	Convolution(m_CNNLayer[0].data, m_CNNLayer[0].Width, m_CNNLayer[0].Height, m_CNNLayer[0].D, 
		    Convparm2, 5, 6, 1, Bparm2,1,1,3,2);

	//block3
	Convolution(m_CNNLayer[1].data, m_CNNLayer[1].Width, m_CNNLayer[1].Height, m_CNNLayer[1].D, 
		    Convparm3, 5, 12, 1, Bparm3,1,2,3,2);

	//block4
	Convolution(m_CNNLayer[2].data, m_CNNLayer[2].Width, m_CNNLayer[2].Height, m_CNNLayer[2].D, 
		     Convparm4, 3, 64, 1, Bparm4,1,0,0,0);

	//outputlayer
	Convolution(m_CNNLayer[3].data, m_CNNLayer[3].Width, m_CNNLayer[3].Height, m_CNNLayer[3].D, 
		     Convparm5, 1, 2, 1, Bparm5,0,0,0,0);

	if(m_CNNLayer[4].data[0][0] > m_CNNLayer[4].data[0][1])
	   label = 1; 
	else
       label = 0; 

	DelCNNLayer();
	return label;
}

/**************************************************************
    name：Convolution
	author：Jane
	brief：同时实现卷积+ReLU+Pooling
	param:
	    input2D：输入数据，维度为inputwidth*inputheight*inputD
		inputwidth：输入矩阵的宽
		inputheight：输入矩阵的高
		inputD：输入矩阵的维度
		size: 卷积核的大小
		f：卷积核，维度为size*size*inputD*featurenum
		featurenum：卷积核个数，即特征数量
        stride：卷积步长
		d = 偏置项
		Relu：是否进行ReLU，0-不进行，1-进行
		Pool：是否进行Pooling，0-不进行，1-平均池化，2-最大池化
		size_pool：池化核大小
		stride_pool：池化步长
**************************************************************/
bool CMyCNN::Convolution(float **input2D, int inputwidth, int inputheight, int inputD, float*f, int size, int featurenum, int stride,
						 float *d, int Relu, int Pool, int size_pool, int stride_pool)
{
	long addsum = 0, mulsum = 0;
	layer layernow;
	layernow.D = featurenum;
	layernow.Width = 1 + (inputwidth - size)/stride;
	layernow.Height= 1 + (inputheight - size)/stride;
	layernow.data = New2DMatrix_double(layernow.D,layernow.Width * layernow.Height);

	long fsizeD2 = inputD*size*size;
	long fsizeD = size*size;

	//转换成可以用下标引用的形式
	//double **input2D = deal_array(input,inputD,inputwidth*inputheight);

	//中间量
	int D2,D,Xstart,Ystart,x,y,XstartStride,YstartStride;

	//#pragma omp parallel	//openmp并行加速
	// XYstartStride：在输入map的位置，XYstart：在输出map的位置
	for(Ystart=0,YstartStride=0; Ystart<layernow.Height; Ystart++,YstartStride=YstartStride+stride)
	{
	  for(Xstart=0,XstartStride=0; Xstart<layernow.Width; Xstart++,XstartStride=XstartStride+stride)
	  {
//#pragma omp parallel for if(isOpenMp)
	    for(D2=0; D2<featurenum; D2++)	//遍历每个卷积核（卷积核的个数）
	    {
		  double temp = 0;
		  for(x=0; x<size; x++)	//卷积核内的位置
		  {
			 for(y=0; y<size; y++)		
		    {
			   for(D=0; D<inputD; D++)	//卷积核的维度
			   {
					double a = input2D[D][XstartStride + x + (YstartStride + y)* inputwidth];
					float b = f[D2*fsizeD2 + D*fsizeD + y*size + x];	//卷积权重值
					temp += input2D[D][XstartStride + x + (YstartStride + y) * inputwidth] * f[D2*fsizeD2 + D*fsizeD + y*size + x];
					addsum += 1;
					mulsum += 1;
			   }
		    } 
		  }
		  layernow.data[D2][Xstart+Ystart*layernow.Width] = temp + d[D2];
		  addsum+=1;

		  // ReLU的实现
		  if (Relu == 1 && layernow.data[D2][Xstart + Ystart*layernow.Width] <= 0)
			  layernow.data[D2][Xstart + Ystart*layernow.Width] = 0;
	    }
	  }
	}

	// Pooling的实现
	if (Pool == 1)	//平均池化
	{
		inputwidth = layernow.Width;
		layernow.Width = 1 + (layernow.Width - size_pool) / stride_pool;
		layernow.Height = 1 + (layernow.Height - size_pool) / stride_pool;
//#pragma omp parallel for if(isOpenMp)
		for (D2 = 0; D2<layernow.D; D2++)	//遍历输入map的每个通道
		{
			for (Ystart = 0, YstartStride = 0; Ystart<layernow.Height; Ystart++, YstartStride = YstartStride + stride_pool)	//卷积窗在原图像中的位置
			{
				for (Xstart = 0, XstartStride = 0; Xstart<layernow.Width; Xstart++, XstartStride = XstartStride + stride_pool)	
				{
					float temp = 0;
#pragma omp parallel for if(isOpenMp)
					for (x = 0; x<size_pool; x++)	//池化核内的位置
					{
						for (y = 0; y<size_pool; y++)
						{
//#pragma omp critical
							//{
							temp += layernow.data[D2][XstartStride + x + (YstartStride + y)*inputwidth];
							addsum += 1;
							//}
							
						}
					}
					layernow.data[D2][Xstart + Ystart*layernow.Width] = temp / (size_pool*size_pool);
					mulsum += 1;
				}
			}
		}
	}
	else if (Pool == 2)	//最大池化
	{
		inputwidth = layernow.Width;
		layernow.Width = 1 + (layernow.Width - size_pool) / stride_pool;
		layernow.Height = 1 + (layernow.Height - size_pool) / stride_pool;
//#pragma omp parallel for if(isOpenMp)
		for (D2 = 0; D2<layernow.D; D2++)
		{
			for (Ystart = 0, YstartStride = 0; Ystart<layernow.Height; Ystart++, YstartStride = YstartStride + stride_pool)//卷积窗在原图像中的位置
			{
				for (Xstart = 0, XstartStride = 0; Xstart<layernow.Width; Xstart++, XstartStride = XstartStride + stride_pool)
				{
					float temp = 0, max = -9999999;
#pragma omp parallel for if(isOpenMp)
					for (x = 0; x<size_pool; x++)	//池化核内的位置
					{
						for (y = 0; y<size_pool; y++)
						{
//#pragma omp critical
							//{
							temp = layernow.data[D2][XstartStride + x + (YstartStride + y)*inputwidth];
							if (temp >= max)
								max = temp;
							//}
							
						}
					}
					layernow.data[D2][Xstart + Ystart*layernow.Width] = max;
				}
			}
		}
	}

	//显示在卷积、ReLU、Pooling过程中的加法和乘法次数
	//std::cout << "加法次数：" << addsum << endl;
	//std::cout << "乘法次数：" << mulsum << endl;

	//显示当前层的信息
	//printf("\nLayer result:\n");
	//DisplayLayer(layernow);
	
	m_CNNLayer.push_back(layernow);	//push_back是STL容器的基本操作，存入的过程
	return 1;
}


/*****************************************
	name：New2DMatrix_double
	author：Jane
	brief：动态创建连续存储2维数组
	param:
		int nRow-高
		int nColumn-宽
*****************************************/
float** CMyCNN::New2DMatrix_double(int nRow, int nColumn)
{
	float** pNew = 0;

	pNew = new float*[nRow];
	pNew[0] = new float [nRow*nColumn];

	for(int j = 1; j < nRow; j++)
	{
		pNew[j]= pNew[0] + j*nColumn;
	}
	memset(pNew[0],0,nRow*nColumn*sizeof(float));
	
	return pNew;
}

//layer结构体释放空间
void CMyCNN::DelCNNLayer()
{
	for(vector<layer>::iterator it = m_CNNLayer.begin(); it != m_CNNLayer.end(); it++)
	{
	  delete [] it->data[0];
	  it->data[0] = NULL;

	  delete [] it->data;
	  it->data = NULL;
	}
	
	//clear是无法释放vector内存的，必须要用swap
	m_CNNLayer.clear();
	vector<layer> temp;
	temp.swap(m_CNNLayer);
}


//显示当前层的信息
void CMyCNN::DisplayLayer(layer l)
{
	int D,Xstart,Ystart=0;
	printf("D = %d\n",l.D);
	printf("width = %d\n",l.Width);
	printf("height = %d\n",l.Height);

	for(D = 0; D<l.D; D++)
	{
      printf("D%d data:\n",D);
	  for(Ystart=0; Ystart<l.Height; Ystart++)
	  {
	   for(Xstart=0; Xstart<l.Width; Xstart++)	//卷积窗在原图像中的位置
	   {
          printf("%.3f ",l.data[D][Xstart+Ystart*l.Width]);
	   }
	   printf("\n");
	  }
	}   
}


//从文本中读取网络参数，初始化网络层的参数
void CMyCNN::ReadParm(string filename,int num)
{
	double featurenum = 0;
	double bandnum = 0;
	double kernelsize = 0;

	ifstream in(filename);
	if(in)
	{
		in>>featurenum;
		in>>bandnum;
		in>>kernelsize;

		int paramnum = (int)featurenum*bandnum*kernelsize*kernelsize;
		float *Convparam = new float [paramnum];
		float *Bparam = new float [(int)featurenum];

		//读取卷积核参数
		for(int i=0; i< paramnum; i++)
           in>>Convparam[i];

		//读取偏置
		for(int i=0; i< featurenum; i++)
           in>>Bparam[i];

		if(num == 1)
		{
			Convparm1 = Convparam;
			Bparm1 = Bparam;
		}
		else if(num == 2)
		{
			Convparm2 = Convparam;
			Bparm2 = Bparam;
		}
		else if(num == 3)
		{
			Convparm3 = Convparam;
			Bparm3 = Bparam;
		}
		else if(num == 4)
		{
			Convparm4 = Convparam;
			Bparm4 = Bparam;
		}
		else
		{
			Convparm5 = Convparam;
			Bparm5 = Bparam;
		}
	}
    else
	{
		printf("open file error!");
	}
}