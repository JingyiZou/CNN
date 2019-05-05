#include"stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include "CnnForwardPass.h"

// openmp����ͷ�ļ�
#include <omp.h>

using namespace std;


/************************************************
	name��ForwardPass
	author��Jane
	brief��ʵ�����򴫲������Ԥ������labelһ��
	param:
		imagedata������ͼ��
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
    name��Convolution
	author��Jane
	brief��ͬʱʵ�־��+ReLU+Pooling
	param:
	    input2D���������ݣ�ά��Ϊinputwidth*inputheight*inputD
		inputwidth���������Ŀ�
		inputheight���������ĸ�
		inputD����������ά��
		size: ����˵Ĵ�С
		f������ˣ�ά��Ϊsize*size*inputD*featurenum
		featurenum������˸���������������
        stride���������
		d = ƫ����
		Relu���Ƿ����ReLU��0-�����У�1-����
		Pool���Ƿ����Pooling��0-�����У�1-ƽ���ػ���2-���ػ�
		size_pool���ػ��˴�С
		stride_pool���ػ�����
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

	//ת���ɿ������±����õ���ʽ
	//double **input2D = deal_array(input,inputD,inputwidth*inputheight);

	//�м���
	int D2,D,Xstart,Ystart,x,y,XstartStride,YstartStride;

	//#pragma omp parallel	//openmp���м���
	// XYstartStride��������map��λ�ã�XYstart�������map��λ��
	for(Ystart=0,YstartStride=0; Ystart<layernow.Height; Ystart++,YstartStride=YstartStride+stride)
	{
	  for(Xstart=0,XstartStride=0; Xstart<layernow.Width; Xstart++,XstartStride=XstartStride+stride)
	  {
//#pragma omp parallel for if(isOpenMp)
	    for(D2=0; D2<featurenum; D2++)	//����ÿ������ˣ�����˵ĸ�����
	    {
		  double temp = 0;
		  for(x=0; x<size; x++)	//������ڵ�λ��
		  {
			 for(y=0; y<size; y++)		
		    {
			   for(D=0; D<inputD; D++)	//����˵�ά��
			   {
					double a = input2D[D][XstartStride + x + (YstartStride + y)* inputwidth];
					float b = f[D2*fsizeD2 + D*fsizeD + y*size + x];	//���Ȩ��ֵ
					temp += input2D[D][XstartStride + x + (YstartStride + y) * inputwidth] * f[D2*fsizeD2 + D*fsizeD + y*size + x];
					addsum += 1;
					mulsum += 1;
			   }
		    } 
		  }
		  layernow.data[D2][Xstart+Ystart*layernow.Width] = temp + d[D2];
		  addsum+=1;

		  // ReLU��ʵ��
		  if (Relu == 1 && layernow.data[D2][Xstart + Ystart*layernow.Width] <= 0)
			  layernow.data[D2][Xstart + Ystart*layernow.Width] = 0;
	    }
	  }
	}

	// Pooling��ʵ��
	if (Pool == 1)	//ƽ���ػ�
	{
		inputwidth = layernow.Width;
		layernow.Width = 1 + (layernow.Width - size_pool) / stride_pool;
		layernow.Height = 1 + (layernow.Height - size_pool) / stride_pool;
//#pragma omp parallel for if(isOpenMp)
		for (D2 = 0; D2<layernow.D; D2++)	//��������map��ÿ��ͨ��
		{
			for (Ystart = 0, YstartStride = 0; Ystart<layernow.Height; Ystart++, YstartStride = YstartStride + stride_pool)	//�������ԭͼ���е�λ��
			{
				for (Xstart = 0, XstartStride = 0; Xstart<layernow.Width; Xstart++, XstartStride = XstartStride + stride_pool)	
				{
					float temp = 0;
#pragma omp parallel for if(isOpenMp)
					for (x = 0; x<size_pool; x++)	//�ػ����ڵ�λ��
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
	else if (Pool == 2)	//���ػ�
	{
		inputwidth = layernow.Width;
		layernow.Width = 1 + (layernow.Width - size_pool) / stride_pool;
		layernow.Height = 1 + (layernow.Height - size_pool) / stride_pool;
//#pragma omp parallel for if(isOpenMp)
		for (D2 = 0; D2<layernow.D; D2++)
		{
			for (Ystart = 0, YstartStride = 0; Ystart<layernow.Height; Ystart++, YstartStride = YstartStride + stride_pool)//�������ԭͼ���е�λ��
			{
				for (Xstart = 0, XstartStride = 0; Xstart<layernow.Width; Xstart++, XstartStride = XstartStride + stride_pool)
				{
					float temp = 0, max = -9999999;
#pragma omp parallel for if(isOpenMp)
					for (x = 0; x<size_pool; x++)	//�ػ����ڵ�λ��
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

	//��ʾ�ھ����ReLU��Pooling�����еļӷ��ͳ˷�����
	//std::cout << "�ӷ�������" << addsum << endl;
	//std::cout << "�˷�������" << mulsum << endl;

	//��ʾ��ǰ�����Ϣ
	//printf("\nLayer result:\n");
	//DisplayLayer(layernow);
	
	m_CNNLayer.push_back(layernow);	//push_back��STL�����Ļ�������������Ĺ���
	return 1;
}


/*****************************************
	name��New2DMatrix_double
	author��Jane
	brief����̬���������洢2ά����
	param:
		int nRow-��
		int nColumn-��
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

//layer�ṹ���ͷſռ�
void CMyCNN::DelCNNLayer()
{
	for(vector<layer>::iterator it = m_CNNLayer.begin(); it != m_CNNLayer.end(); it++)
	{
	  delete [] it->data[0];
	  it->data[0] = NULL;

	  delete [] it->data;
	  it->data = NULL;
	}
	
	//clear���޷��ͷ�vector�ڴ�ģ�����Ҫ��swap
	m_CNNLayer.clear();
	vector<layer> temp;
	temp.swap(m_CNNLayer);
}


//��ʾ��ǰ�����Ϣ
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
	   for(Xstart=0; Xstart<l.Width; Xstart++)	//�������ԭͼ���е�λ��
	   {
          printf("%.3f ",l.data[D][Xstart+Ystart*l.Width]);
	   }
	   printf("\n");
	  }
	}   
}


//���ı��ж�ȡ�����������ʼ�������Ĳ���
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

		//��ȡ����˲���
		for(int i=0; i< paramnum; i++)
           in>>Convparam[i];

		//��ȡƫ��
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