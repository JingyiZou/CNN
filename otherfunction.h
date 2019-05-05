#ifndef _OTHERFUNCTION_H_
#define _OTHERFUNCTION_H_

struct Size
{
		int width;
		int height;
};

extern float** New2DMatrix_float(int nRow, int nColumn);

extern unsigned char** New2DMatrix_u8(int nRow, int nColumn);

extern int** New2DMatrix_int(int nRow, int nColumn);

extern double** New2DMatrix_double(int nRow, int nColumn);

extern void matproduct(double a[],double b[],double c[],int m,int n,int p);

#endif