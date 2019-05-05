#pragma once
#include<vector>
#include<iostream>
using namespace std;
/******************************************
        name:CMyCNN
		author:Jane
		brief:
		   ���׾��������
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

	 //����ͼ��������Ϣ
	 int inputwidth;	//���
	 int inputheight;	//�߶�
	 int inputband;		//ͨ����
	 float **imagedata;	//ͼ������ָ��

	 bool isOpenMp;		//�Ƿ���OpenMP����

	 int ForwardPass(float **imagedata);	//��������Ԥ��

 private:
	  //ÿһ�����ݽṹ��
     typedef struct LAYER
     {
	  float** data;	//���ֵ
	  int Width;    //����Ŀ�
	  int Height;   //����ĸ�
	  int D;        //�����ά��
     }layer;


	 //STL���������ɶ�̬����
	 vector<layer> m_CNNLayer;	//����ÿһ�����Ϣ���ܵĲ���Ϣ

	 bool Convolution(float **input2D,int inputwidth,int inputheight,int inputD,float*f,int size,int featurenum,int stride,float *d,int Relu,int Pool, int size_pool,int stride_pool);	//���������ReLU�������ػ�����
	 float** New2DMatrix_double(int nRow, int nColumn);		//��������ռ�
	 void DelCNNLayer();	//�ͷ����в�Ŀռ�
	 void DisplayLayer(layer l);	//��ʾ��ǰ�����Ϣ
	 void ReadParm(string filename,int num);	//��txt�ж�ȡ�����������ȡ������ʼ����������

	 //���������ƫ�ò���
	 float* Convparm1,* Convparm2,* Convparm3,* Convparm4,* Convparm5;
	 float* Bparm1,* Bparm2,* Bparm3,* Bparm4,* Bparm5;
};