#include "stdafx.h"
#include "otherfunction.h"
#include "stdlib.h"
#include <cstring>
#include "math.h"

const double eps=1e-6;  
  
//合并最终核函数
void matproduct(double a[],double b[],double c[],int m,int n,int p)  
{  
    int i,j,k;  
    for(i=0;i<m;++i)  
    {  
        for(j=0;j<p;++j)  
        {  
            double sum=0;  
            for(k=0;k<n;++k)  
            {  
                sum+=a[i*n+k]*b[k*p+j];  
            }  
            c[i*p+j]=sum;  
        }  
    }

	//show kernel
	/*for(i=0;i<m;++i)  
    {  
		for(j=0;j<p;++j)  
        {  
			printf("%lf\t",c[i*p+j]);  
        }  
        printf("\n");;  
    }*/  
}  


//创建二维数组
float** New2DMatrix_float(int nRow, int nColumn)
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

//创建二维数组
unsigned char** New2DMatrix_u8(int nRow, int nColumn)
{
	unsigned char** pNew = 0;

	pNew = new unsigned char*[nRow];
	pNew[0] = new unsigned char[nRow*nColumn];

	for (int j = 1; j < nRow; j++)
	{
		pNew[j] = pNew[0] + j*nColumn;
	}
	memset(pNew[0], 0, nRow*nColumn*sizeof(unsigned char));

	return pNew;
}

//创建二维数组
double** New2DMatrix_double(int nRow, int nColumn)
{
	double** pNew = 0;

	pNew = new double*[nRow];
	pNew[0] = new double[nRow*nColumn];

	for (int j = 1; j < nRow; j++)
	{
		pNew[j] = pNew[0] + j*nColumn;
	}
	memset(pNew[0], 0, nRow*nColumn*sizeof(double));

	return pNew;
}

//创建二维数组
int** New2DMatrix_int(int nRow, int nColumn)
{
	int** pNew = 0;

	pNew = new int*[nRow];
	pNew[0] = new int[nRow*nColumn];

	for (int j = 1; j < nRow; j++)
	{
		pNew[j] = pNew[0] + j*nColumn;
	}
	memset(pNew[0], 0, nRow*nColumn*sizeof(int));

	return pNew;
}

