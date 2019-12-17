
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <stdio.h>

#define my_ceil(x,y) (x+y-1)/y
#define X_1 32
#define Y_1 13

#define TILE_WIDTH_N 32
#define LOG2_TILE_WIDTH_N 5 //Change % operation if not power of 2!!!
#define TILE_WIDTH_M 32
#define RATIO 1
#define TILE_WIDTH 24
#define K 5
#define ks 25
__constant__ float w_con_dptr[10000];
namespace mxnet
{
namespace op
{

// Compute C = A * B
__global__ void matrixMultiplyShared1(const float * __restrict__ Y, float * __restrict__ Z,
                                     const int B, const int M, const int C, const int H, const int W)
{
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const int numAColumns = ks * C;
  const int numARows = M;
  const int numBRows = numAColumns;
  const int numBColumns = H_out * W_out * B;


  __shared__ float subTileN[Y_1][X_1];


  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = blockIdx.y * Y_1 + ty;
  int Col = blockIdx.x * X_1 + tx;
  float Cvalue = 0;

  int cs = W_out * H_out;
  //int ks = K * K;
  int srow = (Col % (cs)) / W_out;
  int scol = (Col) % W_out; //(Col % (cs)) % W_out;
  int batch = blockIdx.z;

  #define Y4d(i3, i2, i1, i0) Y[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define Z4d(i3, i2, i1, i0) Z[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

  #pragma unroll
  for(int m=0; m < my_ceil(numAColumns,Y_1) ; m++)
  {
	if(m*Y_1+ty<numBRows){
      int tempRow = m*Y_1+ty;
      int channel = tempRow / ks;
      int nrow = srow + (tempRow%ks)/K;
      int ncol = scol + (tempRow%K); //(tempRow%ks)%K;
      subTileN[ty][tx] = Y4d(batch, channel, nrow, ncol);
    }
    else
      subTileN[ty][tx] = 0.0f;
    __syncthreads();
	
	#pragma unroll
    for(int k = 0; k < Y_1; k++)
      if (m*Y_1+k < numAColumns  && Row<numARows)
        Cvalue += w_con_dptr[Row*numAColumns + (m*Y_1+k)] * subTileN[k][tx];
    __syncthreads();
  }

  if(Row<numARows && Col<numBColumns)
    Z4d(batch, Row, srow, scol) = Cvalue;

  #undef Y4d
  #undef Z4d
}

__global__ void matrixMultiplyShared2(const float * __restrict__ X,const float * __restrict__ Y, float * __restrict__ Z,
                                     const int B, const int M, const int C, const int H, const int W)

{

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const int numAColumns = ks * C;
  const int numARows = M;
  const int numBColumns = H_out * W_out * B;

  __shared__ float subTileN[RATIO][TILE_WIDTH_N];
  float registers[TILE_WIDTH_N];
  float rowM[RATIO];
  int Row = threadIdx.y + blockDim.y * blockIdx.y;

  int Col = blockIdx.x*TILE_WIDTH_N;
  int N_col_to_load = Col + threadIdx.y%TILE_WIDTH_N; 

  int cs = W_out * H_out;
  //int ks = K * K;
  int srow = (N_col_to_load % (cs)) / W_out;
  int scol = (N_col_to_load % (cs)) % W_out;
  int batch = blockIdx.z;

  // calculate index of subTileN that thread will load
  int n_row = threadIdx.y>>LOG2_TILE_WIDTH_N; //threadIdx.y/TILE_WIDTH_N;
  int n_col = threadIdx.y & (TILE_WIDTH_N-1); //threadIdx.y%TILE_WIDTH_N;

  //initialize registers to 0
  for(int reg = 0; reg < TILE_WIDTH_N; reg++)
    registers[reg] = 0.0f;

  #define Y4d(i3, i2, i1, i0) Y[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define Z4d(i3, i2, i1, i0) Z[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

  //start iteration loop
  #pragma unroll
  for(int it = 0; it < my_ceil(numAColumns,RATIO); it++)
  {
    //load M
    for(int step = 0; step < RATIO; step++) {
      if (Row<numARows && it*RATIO + step < numAColumns)
        rowM[step] = X[Row*numAColumns + it*RATIO + step];
      else
        rowM[step] = 0.0f;
    }

    //load N
    if(it*RATIO + n_row < numAColumns && N_col_to_load < numBColumns)
    {
      int tempRow = it*RATIO + n_row;
      int channel = tempRow / ks;
      int nrow = srow + (tempRow%ks)/K;
      int ncol = scol + (tempRow%ks)%K;
      subTileN[n_row][n_col] = Y4d(batch, channel, nrow, ncol); // N[(it*RATIO + n_row)*numBColumns + N_col_to_load];
    }
    else
      subTileN[n_row][n_col] = 0.0f;
    __syncthreads();

    #pragma unroll
    for(int step = 0; step < RATIO; step++)
      for(int reg = 0; reg < TILE_WIDTH_N; reg++)
        registers[reg] += rowM[step] * subTileN[step][reg];
    __syncthreads();
  }

  #pragma unroll
  for(int reg = 0; reg < TILE_WIDTH_N; reg++)
  {
    if(Row < numARows && Col + reg < numBColumns)
    {
      int temp_col = Col + reg; 
      int srow = (temp_col % (cs)) / W_out;
      int scol = (temp_col % (cs)) % W_out;
      int batch = blockIdx.z;//temp_col/ cs;
      Z4d(batch, Row, srow, scol) = registers[reg];
    }
  }

  #undef Y4d
  #undef Z4d
}

template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{


    // Extract the tensor dimensions into B,M,C,H,W,K
    
	const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
  //  const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	
    //first layer has 12 output channels and one input channel
   
	//cudaMemcpyToSymbol(w_con_dptr, w.dptr_, M*C*K*K*sizeof(float));//,cudaMemcpyDeviceToDevice); 
	
	if(M==24) {
		dim3 dimBlock(1, TILE_WIDTH_M, 1);
    	dim3 dimGrid(my_ceil(H_out*W_out,TILE_WIDTH_N), my_ceil(M,TILE_WIDTH_M), B);
    	matrixMultiplyShared2<<<dimGrid, dimBlock>>>(w.dptr_, x.dptr_, y.dptr_ , B, M, C, H, W);
	}
	
	else {
		cudaMemcpyToSymbol(w_con_dptr, w.dptr_, M*C*ks*sizeof(float));//,cudaMemcpyDeviceToDevice); 
		dim3 dimBlock(X_1, Y_1, 1);
    	dim3 dimGrid(my_ceil(H_out*W_out,X_1), my_ceil(M,Y_1), B);
    	matrixMultiplyShared1<<<dimGrid, dimBlock>>>(x.dptr_, y.dptr_ , B, M, C, H, W);
	}		

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    //MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    // CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
