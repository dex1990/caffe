#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define THREAD_NUM 1024
int thread_num = 512;
cudaError_t cvtWithCuda(float *c,  unsigned char *a, int size,int idx);
cudaError_t clipWithCuda(float *c,  unsigned char *a, int size,int idx);
__global__ void cvtKernel(float *c,  unsigned char *a,int size,int tdnum)  
{        
        const int tid = threadIdx.x;
        int i;
        for (i = tid; i < size; i += tdnum){
            c[i] = (float)a[i]/255.0;  
        }
} 

__global__ void clipKernel(float *c,  unsigned char *a,int size,int tdnum)  
{        
        const int tid = threadIdx.x;
        int i;
        float res;
        for (i = tid; i < size; i += tdnum){
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
    thread_num = prop.maxThreadsPerBlock;
//    printf("ThreadsperBlock %d\n",thread_num); 
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
 
//    printf("before call kernel\n"); 
    // Launch a kernel on the GPU with one thread for each element.  
//    dim3 blockk(16,16);
//    dim3 threadd(16,16);
//    cvtKernel<<<blockk, threadd>>>(dev_c, dev_a);
    cvtKernel<<<1, thread_num>>>(dev_c, dev_a,size,thread_num);  
//    printf("after call kernel\n");
    // cudaThreadSynchronize waits for the kernel to finish, and returns  
    // any errors encountered during the launch.  
    cudaStatus = cudaThreadSynchronize(); 
//    printf("call cudaThreadSynchronize\n"); 
    if(cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);  
        goto Error;  
    }  
  
    // Copy output vector from GPU buffer to host memory.  
/*    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMemcpy failed!");  
        goto Error;  
    }  
*/
Error:  
//    cudaFree(dev_c);  
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
 
//    printf("before call clipkernel\n"); 
    // Launch a kernel on the GPU with one thread for each element.  
    clipKernel<<<1, thread_num>>>(dev_c, dev_a,size,thread_num);  
//    printf("after call clipkernel\n");
    // cudaThreadSynchronize waits for the kernel to finish, and returns  
    // any errors encountered during the launch.  
    cudaStatus = cudaThreadSynchronize(); 
//    printf("call cudaThreadSynchronize\n"); 
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
//    cudaFree(dev_c);  
    cudaFree(dev_a);  
      
    return cudaStatus; 
}

extern "C" int Cudacvt(float *c,unsigned char *a,int w,int h,int idx){
    // Add vectors in parallel.  
    cudaError_t cudaStatus;  
    /*int num = 0;  
    cudaDeviceProp prop;  
    cudaStatus = cudaGetDeviceCount(&num);  
    for(int i = 0;i<num;i++)  
    {  
        cudaGetDeviceProperties(&prop,i);
         
    } */ 
    cudaStatus = cvtWithCuda(c, a,w*h,idx);  
    if (cudaStatus != cudaSuccess)   
    {  
        fprintf(stderr, "addWithCuda failed!");  
        return -1;  
    }  
    // cudaThreadExit must be called before exiting in order for profiling and  
    // tracing tools such as Nsight and Visual Profiler to show complete traces.  
//    cudaStatus = cudaThreadExit();  
//    if (cudaStatus != cudaSuccess)   
//    {  
//        fprintf(stderr, "cudaThreadExit failed!");  
//        return -1;  
//    }  
    return 0;
}
extern "C" int Cudaclip(float *c,unsigned char *a,int w,int h,int idx){
    // Add vectors in parallel.  
    cudaError_t cudaStatus;  
    /*int num = 0;  
    cudaDeviceProp prop;  
    cudaStatus = cudaGetDeviceCount(&num);  
    for(int i = 0;i<num;i++)  
    {  
        cudaGetDeviceProperties(&prop,i);
         
    } */ 
    cudaStatus = clipWithCuda(c,a,w*h,idx);  
    if (cudaStatus != cudaSuccess)   
    {  
        fprintf(stderr, "addWithCuda failed!");  
        return -1;  
    }  
    // cudaThreadExit must be called before exiting in order for profiling and  
    // tracing tools such as Nsight and Visual Profiler to show complete traces.  
//    cudaStatus = cudaThreadExit();  
//    if (cudaStatus != cudaSuccess)   
//    {  
//        fprintf(stderr, "cudaThreadExit failed!");  
//        return -1;  
//    }  
    return 0;
}
/*  
int main() {
    printf("start\n");
    unsigned char a[10000];
    float c[10000];
    memset(a, 0xff, 10000*sizeof(unsigned char));
    printf("a[0] : %d\n",a[0]);
    Cudacvt(c, a, 100, 100);
    return 0;
}
*/
