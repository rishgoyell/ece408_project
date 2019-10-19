
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

# define TILE_WIDTH 16		// We will use 4 for small examples.

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a



namespace mxnet
{
namespace op
{

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
        for (int c = 0;  c < C; c++) {		// sum over all input channels
            for (int p = 0; p < K; p++)		// loop over KxK  filter
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
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working



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
    
    int W_grid = ceil(W_out*1.0/TILE_WIDTH); 	// number of horizontal tiles per output map
    int H_grid = ceil(H_out*1.0/TILE_WIDTH); 	// number of vertical tiles per output map
    int Y = H_grid * W_grid;			// Number of blocks in the Y dimension
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, Y, B);

    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

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