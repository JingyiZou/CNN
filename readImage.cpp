#include"readImage.h"

//Mat转uchar指针
float** Mat2pChar(const Mat& img)
{
	int width = img.cols, height = img.rows;
	int band = img.channels();
	//开辟对应的图像数据存储内存
	float** pImg = new float*[band];
	for(int i=0;i<band;i++){
		pImg[i] = new float[width*height];
		memset(pImg[i], 0, sizeof(float)*width*height);
	}
	int nr = height;
	//将Mat对象中的具体图像内存值赋值给二维指针
	for (int j=0; j<nr; j++) {
		  const uchar* data= img.ptr<uchar>(j);
		  for (int i=0; i<width; i++) {
			  for(int b=0;b<band;b++){
				  auto c = data[i*band+b];
				  pImg[b][j*width+i]=float(c);
			  }
          }                 
     }
	return pImg;
}


//uchar指针转Mat
Mat pChar2Mat(unsigned char** pImg, int width, int height, int band){
	Mat mm;
	if(band==1)
		mm.create(width, height, CV_8UC1);
	else if(band == 3)
		mm.create(width, height, CV_8UC3);
	int nr = height;
	int nc = width *band;
	for (int j=0; j<nr; j++) {
		  uchar* data= mm.ptr<uchar>(j);
		  for (int i=0; i<width; i++) {
			  for(int b=0;b<band;b++){
				  data[i*band+b]= pImg[b][j*width+i];
			  }
          }                 
     }
	return mm;
}

