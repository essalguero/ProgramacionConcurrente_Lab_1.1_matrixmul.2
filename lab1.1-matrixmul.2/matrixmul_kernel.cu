/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

    __global__ void
matrixMul(
    float* P, const float* M, const float* N,
    const int Mh, const int Mw, const int Nw,
    const int block_size)
{
        const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float Psub = 0;
    int i = 0, indexM = 0, indexN = 0, indexP = 0;

    // ===================================================================
    // Code Segment 5
    // Determine the output index of each thread.
    // Compute the dot product of one row of M and one column of N
    // for each thread.
    // Write the computed value to matrix P at the correct index.
    // ===================================================================
    int indX = bx * blockDim.x + tx;
    int indY = by * blockDim.y + ty;

    int nElementosFila = gridDim.x * blockDim.x;

    /*if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1 && blockIdx.y == 0)
	printf("indX: %d, indY: %d, nElementosFila: %d\n", indX, indY, nElementosFila);
    */
    for (i = 0; i < Mw; ++i)
    {
//Psub += M[indX][i] * N[i][indY];

        indexM = (indX * nElementosFila) + i;
        indexN = indY + (nElementosFila * i);

        Psub += M[indexM] * N[indexN];

        indexP = (indX * gridDim.x * blockDim.x) + indY;

        /*if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1 && blockIdx.y == 0)
	    //printf("Indexes -> %d %d: %d - %d\n", blockIdx.x * block_size + i, blockIdx.y * block_size + i, indexM, indexN);
	    printf("bx: %d, by: %d, tx: %d, ty: %d, indX: %d, intY: %d, indexP: %d, indexM: %d, indexN: %d, M[indexM]: %f, N[indexN]: %f, Psub: %f\n", bx, by, tx, ty, indX, indY, indexP, indexM, indexN, M[indexM], N[indexN], Psub);*/
    }


    P[indexP] = Psub;

	//printf("Result -> Psub: %f -> P[indexP]: %f\n\n\n", P[indexP]);

    // End of Code Segment 5 ============================================
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_


