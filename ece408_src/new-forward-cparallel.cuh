
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <stdio.h>

#define TILE_WIDTH 32
#define CBLOCKS 12

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a



namespace mxnet
{
namespace op
{

// Compute C = A * B
__global__ void matrixMultiplyShared(const float *  __restrict__ X, const float * __restrict__ Y, float * __restrict__ Z,
                                     const int B, const int M, const int C, const int H, const int W, const int K)
{
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  int numAColumns = K * K * ((C-1)/CBLOCKS + 1);
  int numARows = M;
  int numBRows = numAColumns;
  int numBColumns = H_out * W_out * B;

  int numCRows = numARows;
  int numCColumns = numBColumns;

  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];


  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bz = blockIdx.z;

  int Row = blockIdx.y * TILE_WIDTH + ty;
  int Col = blockIdx.x * TILE_WIDTH + tx;
  float Cvalue = 0;

  int cs = W_out * H_out;
  int ks = K * K;
  int srow = (Col % cs) / W_out;
  int scol = (Col % cs) % W_out;
  int batch = Col / cs;

  #define Y4d(i3, i2, i1, i0) Y[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define Z4d(i3, i2, i1, i0) Z[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

  #pragma unroll
  for(int m=0; m<ceil((1.0*numAColumns)/TILE_WIDTH) ; m++)
  {
    int channelIdx = bz + CBLOCKS * ((m*TILE_WIDTH+tx)/ks);
    // int channelIdx = ceil(float(C)/CBLOCKS) * tz + ((m*TILE_WIDTH+tx)/ks);
    // printf("W %d %d %d", tz, tx, channelIdx);
    if(Row<numARows && m*TILE_WIDTH+tx<numAColumns && channelIdx<C)
      subTileM[ty][tx]=X[Row*ks*C + channelIdx*ks + (m*TILE_WIDTH+tx)%ks];
    else
      subTileM[ty][tx]=0;

    channelIdx = bz + CBLOCKS * ((m*TILE_WIDTH+ty)/ks);
    // channelIdx = ceil(float(C)/CBLOCKS) * tz + ((m*TILE_WIDTH+ty)/ks);
    // printf("I %d %d %d", tz, ty, channelIdx);
    if(Col<numCColumns && m*TILE_WIDTH+ty<numBRows && channelIdx<C){
      int tempRow = m*TILE_WIDTH+ty;
      subTileN[ty][tx] = Y4d(batch, channelIdx, srow+(tempRow%ks)/K, scol+(tempRow%ks)%K);
    }
    else
      subTileN[ty][tx] = 0;

    __syncthreads();
    #pragma unroll
    for(int k = 0; k < TILE_WIDTH; k++)
      if (m*TILE_WIDTH+k < numAColumns)
        Cvalue += subTileM[ty][k] * subTileN[k][tx];
    __syncthreads();
  }

  if(Row<numCRows && Col<numCColumns)
    atomicAdd(&Z4d(batch, Row, srow, scol), Cvalue);
    // Z4d(batch, Row, srow, scol) += Cvalue;

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
    printf("B:%d, M:%d, C:%d, H:%d, W:%d, K:%d, CBLOCKS:%d", B,M,C,H,W,K,CBLOCKS);

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;



    // float* x_cpu = (float *) malloc(sizeof(float) * B*C*H*W);
    // cudaMemcpy ( x_cpu, x.dptr_, sizeof(float) * B*C*H*W, cudaMemcpyDeviceToHost );

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(ceil((1.0 * B * H_out*W_out)/TILE_WIDTH), ceil((1.0*M)/TILE_WIDTH), CBLOCKS);
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(w.dptr_, x.dptr_, y.dptr_ , B, M, C, H, W, K);

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

