#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define THREAD_NUM 512
#define BLOCK_NUM 32

cudaError_t cvtWithCuda(float *c,  unsigned char *a, int size,int idx);
cudaError_t clipWithCuda(float *c,  unsigned char *a, int size,int idx);
__global__ void cvtKernel(float *c,  unsigned char *a,int size)  
{        
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        int i;
        for (i = bid * THREAD_NUM + tid; i < size; i += THREAD_NUM * BLOCK_NUM){
                    c[i] = (float)a[i]/255.0;
        }
} 

__global__ void clipKernel(float *c,  unsigned char *a,int size)  
{        
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        int i;
        float res;
        for (i = bid * THREAD_NUM + tid; i < size; i += THREAD_NUM * BLOCK_NUM){     
            res =(float)a[i] - c[i]*255.0;
            if(res < 0)
                res = 0;
            if(res > 255)
                res = 255;
            a[i] = (unsigned char)res;
        }
} 

cudaError_t cvtWithCuda(float *c, unsigned char*a, int  size,int idx)  
{  
    unsigned char *dev_a = 0; 
    float *dev_c = c;  
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system. 
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,idx);
    cudaStatus = cudaSetDevice(idx);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");  
        goto Error;  
    }  

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(unsigned char));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        goto Error;  
    }  
      
    // Copy input vectors from host memory to GPU buffers.  
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(unsigned char), cudaMemcpyHostToDevice);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMemcpy failed!");  
        goto Error;  
    }  
 
    // Launch a kernel on the GPU with one thread for each element.  
    cvtKernel<<<BLOCK_NUM, THREAD_NUM>>>(dev_c, dev_a,size);  
    cudaStatus = cudaThreadSynchronize(); 
    if(cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);  
        goto Error;  
    }  
Error:  
    cudaFree(dev_a);  
      
    return cudaStatus;  
}

cudaError_t clipWithCuda(float *c, unsigned char*a,int size,int idx)  
{  
    unsigned char *dev_a = 0;  
    float *dev_c = c;  
    cudaError_t cudaStatus; 
    // Choose which GPU to run on, change this on a multi-GPU system.  
    cudaStatus = cudaSetDevice(idx);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");  
        goto Error;  
    }  
  
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(unsigned char));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        goto Error;  
    }  
    // Copy input vectors from host memory to GPU buffers.  
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(unsigned char), cudaMemcpyHostToDevice);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMemcpy failed!");  
        goto Error;  
    }  
 
    // Launch a kernel on the GPU with one thread for each element.  
    clipKernel<<<BLOCK_NUM, THREAD_NUM>>>(dev_c, dev_a,size);  
    cudaStatus = cudaThreadSynchronize(); 
    if(cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching clipKernel!\n", cudaStatus);  
        goto Error;  
    }  
  
    // Copy output vector from GPU buffer to host memory.  
    cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMemcpy failed!");  
        goto Error;  
    }  

Error:  
    cudaFree(dev_a);  
      
    return cudaStatus; 
}

extern "C" int Cudacvt(float *c,unsigned char *a,int w,int h,int idx){
    cudaError_t cudaStatus;  
    cudaStatus = cvtWithCuda(c, a,w*h,idx);  
    if (cudaStatus != cudaSuccess)   
    {  
        fprintf(stderr, "addWithCuda failed!");  
        return -1;  
    }  
    return 0;
}
extern "C" int Cudaclip(float *c,unsigned char *a,int w,int h,int idx){
    // Add vectors in parallel.  
    cudaError_t cudaStatus;  
    cudaStatus = clipWithCuda(c,a,w*h,idx);  
    if (cudaStatus != cudaSuccess)   
    {  
        fprintf(stderr, "addWithCuda failed!");  
        return -1;  
    }  
    return 0;
}
