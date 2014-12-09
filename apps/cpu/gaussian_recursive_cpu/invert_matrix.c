/**
 * \file invert_matrix.c
 * \brief Invert matrix through QR decomposition
 * \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
 *
 * Copyright (c) 2012-2013, Pascal Getreuer
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under, at your option, the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version, or the terms of the
 * simplified BSD license.
 *
 * You should have received a copy of these licenses along with this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include "invert_matrix.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

/**
 * \brief Invert matrix through QR decomposition
 * \param inv_A pointer to memory for holding the result
 * \param A pointer to column-major matrix data
 * \param N the number of dimensions
 * \return 1 on success, 0 on failure
 *
 * The input data is overwritten during the computation. \c inv_A
 * should be allocated before calling this function with space for at least
 * N^2 doubles. Matrices are represented in column-major format, meaning
 *    A(i,j) = A[i + N*j], 0 <= i, j < N.
 */
int invert_matrix(double *inv_A, double *A, int N)
{
    double *c = NULL, *d = NULL, *col_j, *col_k, *inv_col_k;
    double temp, scale, sum;
    int i, j, k, success = 0;
    
    assert(inv_A && A && N > 0);
    
    if (!(c = (double *)malloc(sizeof(double) * N))
        || !(d = (double *)malloc(sizeof(double) * N)))
        goto fail;
    
    for (k = 0, col_k = A; k < N - 1; ++k, col_k += N)
    {
        scale = 0.0;
        
        for (i = k; i < N; ++i)
            if ((temp = fabs(col_k[i])) > scale)
                scale = temp;
        
        if (scale == 0.0)
            goto fail; /* Singular matrix */
        
        for (i = k; i < N; ++i)
            col_k[i] /= scale;
        
        for (sum = 0.0, i = k; i < N; ++i)
            sum += col_k[i]*col_k[i];
        
        temp = (col_k[k] >= 0.0) ? sqrt(sum) : -sqrt(sum);
        col_k[k] += temp;
        c[k] = temp * col_k[k];
        d[k] = -scale * temp;
        
        for (j = k + 1, col_j = col_k + N; j < N; ++j, col_j += N)
        {
            for (scale = 0.0, i = k; i < N; ++i)
                scale += col_k[i] * col_j[i];
                
            scale /= c[k];
            
            for (i = k; i < N; ++i)
                col_j[i] -= scale * col_k[i];
        }
    }

    d[N-1] = col_k[k];
    
    if (d[N - 1] == 0.0)
        goto fail; /* Singular matrix */
    
    for (k = 0, inv_col_k = inv_A; k < N; ++k, inv_col_k += N)
    {
        for (i = 0; i < N; ++i)
            inv_col_k[i] = -A[k] * A[i] / c[0];
        
        inv_col_k[k] += 1.0;
        
        for (j = 1, col_j = A + N; j < N-1; ++j, col_j += N)
        {
            for (scale = 0.0, i = j; i < N; ++i)
                scale += col_j[i] * inv_col_k[i];
            
            scale /= c[j];
            
            for (i = j; i < N; ++i)
                inv_col_k[i] -= scale * col_j[i];
        }
        
        inv_col_k[j] /= d[N-1];
        
        for (i = N - 2; i >= 0; --i)
        {
            for (sum = 0.0, j = i + 1, col_j = A + N*(i + 1);
                j < N; ++j, col_j += N)
                sum += col_j[i] * inv_col_k[j];
            
            inv_col_k[i] = (inv_col_k[i] - sum) / d[i];
        }
    }
    
    success = 1; /* Finished successfully */
fail: /* Clean up */
    free(d);
    free(c);
    return success;
}
