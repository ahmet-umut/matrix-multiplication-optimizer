/********************************************************
 * Kernels to be optimized for the CS:APP Performance Lab
 ********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
//#include <immintrin.h>
#include <math.h>
/*
 * Please fill in the following team_t struct
 */
team_t team = {

		"e2448066",      /* First student ID */
		"Ahmet Umut",       /* First student name */

};


/********************
 * NORMALIZATION KERNEL
 ********************/

/****************************************************************
 * Your different versions of the normalization functions go here
 ***************************************************************/

/*
 * naive_normalize - The naive baseline version of convolution
 */
char naive_normalize_descr[] = "naive_normalize: Naive baseline implementation";
void naive_normalize(int dim, float *src, float *dst) {
	float min, max;
	min = src[0];
	max = src[0];

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
	
			if (src[RIDX(i, j, dim)] < min) {
				min = src[RIDX(i, j, dim)];
			}
			if (src[RIDX(i, j, dim)] > max) {
				max = src[RIDX(i, j, dim)];
			}
		}
	}

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			dst[RIDX(i, j, dim)] = (src[RIDX(i, j, dim)] - min) / (max - min);
		}
	}
}

/*
 * normalize - Your current working version of normalization
 * IMPORTANT: This is the version you will be graded on
 */
char normalize_descr[] = "Normalize: Current working version";
void normalize(int dim, float *src, float *dst)
{
	float(*srcmatr)[dim] = (float(*)[dim]) src; // cast src to 2D array
	float(*dstmatr)[dim] = (float(*)[dim]) dst; // cast dst to 2D array

	float min, max;
	min = src[0];
	max = src[0];

/* 	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
	
			//if (src[RIDX(i, j, dim)] < min)
			if (srcmatr[i][j] < min)
				min = src[RIDX(i, j, dim)];
			//else if (src[RIDX(i, j, dim)] > max)
			else if (srcmatr[i][j] > max)
				max = src[RIDX(i, j, dim)];
		} 
	} */
	for (float(*row)[dim]=srcmatr; row<srcmatr+dim; row++)
	{
		float (* const rowdim) = *row+dim;
		for (float(*cell)=*row; cell<rowdim; cell+=2)
		{
			if (cell[0] < min)	min = cell[0];
			else if (cell[0] > max)	max = cell[0];
			if (cell[1] < min)	min = cell[1];
			else if (cell[1] > max)	max = cell[1];
		}
	}

	/* for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			//dst[RIDX(i, j, dim)] = (src[RIDX(i, j, dim)] - min) / (max - min);
			dstmatr[i][j] = (srcmatr[i][j] - min) / (max - min);
		}
	} */
	const float range = max-min;
	for (float(*dstrow)[dim]=dstmatr, (*srcrow)[dim]=srcmatr; dstrow<dstmatr+dim; dstrow++, srcrow++)
	{
		float(* const value1) = *dstrow+dim;
		for (float(*dstcell)=*dstrow, (*srccell)=*srcrow; dstcell<value1; dstcell++,srccell++)
		{
			*dstcell = (*srccell-min) / range;
		}
	}
}

/*********************************************************************
 * register_normalize_functions - Register all of your different versions
 *     of the normalization functions  with the driver by calling the
 *     add_normalize_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.
 *********************************************************************/

void register_normalize_functions() {
	//add_normalize_function(&naive_normalize, naive_normalize_descr);
	add_normalize_function(&normalize, normalize_descr);
	/* ... Register additional test functions here */
}




/************************
 * KRONECKER PRODUCT KERNEL
 ************************/

/********************************************************************
 * Your different versions of the kronecker product functions go here
 *******************************************************************/

/*
 * naive_kronecker_product - The naive baseline version of k-hop neighbours
 */
char naive_kronecker_product_descr[] = "Naive Kronecker Product: Naive baseline implementation";
void naive_kronecker_product(int dim1, int dim2, float *mat1, float *mat2, float *prod)
{
	for (int i = 0; i < dim1; i++) {
		for (int j = 0; j < dim1; j++) {
			for (int k = 0; k < dim2; k++) {
				for (int l = 0; l < dim2; l++) {
					prod[RIDX(i, k, dim2) * (dim1 * dim2) + RIDX(j, l, dim2)] = mat1[RIDX(i, j, dim1)] * mat2[RIDX(k, l, dim2)];
				}
			}
		}
	}
}

/*
 * kronecker_product - Your current working version of kronecker_product
 * IMPORTANT: This is the version you will be graded on
 */
char kronecker_product_descr[] = "Kronecker Product: Current working version";
void kronecker_product(int dim1, int dim2, float *mat1, float *mat2, float *prod)
{
	float(* matrix1)[dim1] = (float(*)[dim1]) mat1; // cast mat1 to 2D array
	float(* matrix2)[dim2] = (float(*)[dim2]) mat2; // cast mat2 to 2D array
	float(* const product)[dim1 * dim2] = (float(*)[dim1 * dim2]) prod; // cast prod to 2D array

	for (float(*i)[dim1] = matrix1; i < matrix1 + dim1; i++)
	{
		const int value1 = dim2 * (i-matrix1);
		//for (int j = 0; j < dim1; j++)
		float(* const value9) = *i+dim1;
		int value8 = 0;
		for (float(*j) = *i; j < value9; j++)
		{
			int value2 = value8++ * dim2;
			const float value4 = *j;
			//for (int k = 0; k < dim2; k++)
			int value7 = value1;
			for (float(*k)[dim2] = matrix2; k < matrix2 + dim2; k++)
			{
				//const int productrow = k-matrix2 + value1;
				float(* productrow) = product[value7++];
				const float (* const value5) = *k+dim2;
				//for (int l = 0; l < dim2; l++)
				int value6 = value2;

				/* #pragma GCC unroll 4
				for (float(*l) = *k; l < value5; l++)
				{
					//prod[RIDX(i, k, dim2) * (dim1 * dim2) + RIDX(j, l, dim2)] = mat1[RIDX(i, j, dim1)] * mat2[RIDX(k, l, dim2)];
					//product[i * dim2 + k][j * dim2 + l] = matrix1[i][j] * matrix2[k][l];
					//product[(i - matrix1) * dim2 + (k - matrix2)][(j - *i) * dim2 + (l - *k)] = *j * *l;
					productrow[value6++] = value4 * *l;
				} */
				for (float(*l) = *k; l < value5; l+=4, value6+=4)
				{
					productrow[value6] = value4 * l[0];
					productrow[value6+1] = value4 * l[1];
					productrow[value6+2] = value4 * l[2];
					productrow[value6+3] = value4 * l[3];
				}
			}
		}
	}
}

/******************************************************************************
 * register_kronecker_product_functions - Register all of your different versions
 *     of the kronecker_product with the driver by calling the
 *     add_kronecker_product_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.  
 ******************************************************************************/

void register_kronecker_product_functions() {
	//add_kronecker_product_function(&naive_kronecker_product, naive_kronecker_product_descr);
	//add_kronecker_product_function(&kronecker_product, kronecker_product_descr);
	/* ... Register additional test functions here */
}