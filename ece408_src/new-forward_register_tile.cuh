
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <stdio.h>

// #define BLOCK_SIZE 64   // We will use 4 for small examples.
#define TILE_WIDTH 16

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a


#define TILE_WIDTH_N 16
#define TILE_WIDTH_M 64
#define RATIO 4
namespace mxnet
{
namespace op
{

// Compute C = A * B
__global__ void matrixMultiplyShared(const float * __restrict__ X, const float * __restrict__ Y, float * __restrict__ Z,
                                     const int B, const int M, const int C, const int H, const int W, const int K, const int numARows, const int numAColumns, const int numBColumns)
{
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;


  __shared__ float subTileN[RATIO][TILE_WIDTH_N];
  float registers[TILE_WIDTH_N];
  float rowM[RATIO];
  int Row = threadIdx.y + blockDim.y * blockIdx.y;

  int Col = blockIdx.x*TILE_WIDTH_N;
  int N_col_to_load = Col + threadIdx.y%TILE_WIDTH_N; 

  int cs = W_out * H_out;
  int ks = K * K;
  int srow = (N_col_to_load % (cs)) / W_out;
  int scol = (N_col_to_load % (cs)) % W_out;
  int batch = N_col_to_load / cs;

  // calculate index of subTileN that thread will load
  int n_row = threadIdx.y/TILE_WIDTH_N;
  int n_col = threadIdx.y%TILE_WIDTH_N;

  //initialize registers to 0
  for(int reg = 0; reg < TILE_WIDTH_N; reg++)
    registers[reg] = 0.0;

  #define Y4d(i3, i2, i1, i0) Y[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define Z4d(i3, i2, i1, i0) Z[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

  //start iteration loop
  #pragma unroll
  for(int it = 0; it < ceil(numAColumns*1.0/RATIO); it++)
  {
    //load M
    for(int step = 0; step < RATIO; step++) {
      if (Row<numARows && it*RATIO + step < numAColumns)
        rowM[step] = X[Row*numAColumns + it*RATIO + step];
      else
        rowM[step] = 0.0;
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
      subTileN[n_row][n_col] = 0.0;
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
      int batch = temp_col / cs;
      Z4d(batch, Row, srow, scol) = registers[reg];
    }
  }

  #undef Y4d
  #undef Z4d
}


    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

__global__ void unroll(float * __restrict__ X_out, const float * __restrict__ X, const int M, const int C, const int H, const int W, const int K, const int size){
    // linearized idx across images
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx >= size)
        return;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define x4d(i2, i1, i0) X[(i2) * (H * W) + (i1) * (W) + i0]

    // wrong because idx is across images, get idx within image.
    // const int idx_within_img = idx % (C * K * K * H_out * W_out);
    int row = idx / (H_out * W_out);
    const int col = idx % (H_out * W_out);
    int col_increment = row % K;
    row /= K;
    int row_increment = row % K;
    int channel = row / K;
    int col_within_channel_start = col % W_out;
    int row_within_channel_start = col / W_out;
    X_out[idx] = x4d(channel, row_within_channel_start + row_increment, col_within_channel_start+col_increment);
}
#undef x4d


__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int num_tiles_per_block_horizontal = ceil(W_out*1.0/TILE_WIDTH);

    int m = blockIdx.x;
    int th = blockIdx.y / num_tiles_per_block_horizontal ;
    int tw = blockIdx.y % num_tiles_per_block_horizontal ;

    int h = th*TILE_WIDTH + threadIdx.y;
    int w = tw*TILE_WIDTH + threadIdx.x;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    float acc = 0;
    if (h < H_out && w < W_out) {
        for (int c = 0;  c < C; c++) {    // sum over all input channels
            for (int p = 0; p < K; p++)   // loop over KxK  filter
                for (int q = 0; q < K; q++)
                    acc += x4d(blockIdx.z, c, h + p, w + q) * k4d(m, c, p, q);
        }
        y4d(blockIdx.z, m, h, w) = acc;
    }
}

    #undef y4d
    #undef x4d
    #undef k4d

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;



    // float* x_cpu = (float *) malloc(sizeof(float) * B*C*H*W);
    // cudaMemcpy ( x_cpu, x.dptr_, sizeof(float) * B*C*H*W, cudaMemcpyDeviceToHost );

    dim3 dimBlock(1, TILE_WIDTH_M, 1);
    dim3 dimGrid( ceil((1.0 * B * H_out*W_out)/TILE_WIDTH_N), ceil((1.0*M)/TILE_WIDTH_M), 1);
    int numAColumns = K * K * C;
    int numARows = M;
    int numBColumns = H_out * W_out * B;
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(w.dptr_, x.dptr_, y.dptr_ , B, M, C, H, W, K, numARows, numAColumns, numBColumns);

    // free(X_unrolled_cpu);
    // free(x_cpu);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
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
