
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <stdio.h>

#define BLOCK_SIZE 64   // We will use 4 for small examples.
#define TILE_WIDTH 16

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a



namespace mxnet
{
namespace op
{

// Compute C = A * B
__global__ void matrixMultiplyShared(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C,
                                     const int numARows, const int numAColumns,
                                     const int numBRows, const int numBColumns)
{

  // B += blockIdx.z * numBRows * numBColumns;
  // C += blockIdx.z * numARows * numBColumns;

  int numCRows = numARows;
  int numCColumns = numBColumns;
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Cvalue = 0;

  #pragma unroll
  for(int m=0; m < ceil(1.0*numAColumns/TILE_WIDTH) ; m++)
  {
    if(Row<numARows && m*TILE_WIDTH+tx<numAColumns)
      subTileM[ty][tx]=A[Row*numAColumns + (m*TILE_WIDTH+tx)];
    else
      subTileM[ty][tx]=0;

    if(m*TILE_WIDTH+ty<numBRows)
      subTileN[ty][tx]=B[numBColumns*(m*TILE_WIDTH+ty) + Col];
    else
      subTileN[ty][tx]=0;

    __syncthreads();
    #pragma unroll
    for(int k = 0; k < TILE_WIDTH; k++)
      if (m*TILE_WIDTH+k < numAColumns)
        Cvalue += subTileM[ty][k] * subTileN[k][tx];
    __syncthreads();
  }
  if(Row<numCRows && Col<numCColumns){
    C[Row*numCColumns+Col] = Cvalue;
  }
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

    float* X_unrolled;
    int size =  C * K * K * H_out * W_out;
    cudaMalloc(&X_unrolled, sizeof(float) * size);

    for(int i = 0; i<B; i++) {

      // fprintf(fp, "B %d M %d C%d H%d W%d K%d \n", B, M, C, H, W, K);

      // #define x4d(i3, i2, i1, i0) x_cpu[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]


      // fprintf(fp, "Printing the first input image\n");
      // for(int i = 0; i < B; i++) {
      //   for(int j = 0; j < C; j++) {
      //     for(int k = 0; k < H; k++){
      //       for(int l = 0; l < W; l++)
      //         fprintf(fp, "%f ", x4d(i, j, k, l));
      //       fprintf(fp, "\n");
      //     }
      //     fprintf(fp, "\n");
      //   }
      //   fprintf(fp, "\n\n\n");
      // }

      // #undef x4d
      // Call the unroll kernel
      dim3 blockDim(BLOCK_SIZE, 1, 1);
      dim3 gridDim(ceil(1.0*size/BLOCK_SIZE), 1, 1);
      unroll<<<gridDim, blockDim>>>(X_unrolled, x.dptr_ + i*C*H*W, M, C, H, W, K, size);

      // fprintf(fp, "Printing first unroll\n");
      // float* X_unrolled_cpu = (float *) malloc(sizeof(float) * size);
      // cudaMemcpy ( X_unrolled_cpu, X_unrolled, size* sizeof(float), cudaMemcpyDeviceToHost );
      // for(int i =0; i <  size; i++){
      //   fprintf(fp, "%f ", X_unrolled_cpu[i]);
      // }

      // Call the matrix multiplication kernel
      dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
      dim3 dimGrid(ceil((1.0*H_out*W_out)/TILE_WIDTH), ceil((1.0*M)/TILE_WIDTH), 1);
      matrixMultiplyShared<<<dimGrid, dimBlock>>>(w.dptr_, X_unrolled, y.dptr_ + i*M*H_out*W_out, M, C*K*K, C*K*K, H_out*W_out);

    }

    cudaFree(X_unrolled);

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
