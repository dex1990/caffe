#include <pthread.h>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#define MAXDEVICENUM 16

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

struct PthreadParams{
    int idx;
    int gpuidx;
    int netidx;
    int step;
    int n;
    int c;
    int w;
    int h;
    unsigned char *data;
    char *prototxt;
    char *model;
    int setdata;
};

class Logger : public ILogger
{
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override
  {
     if (severity != Severity::kINFO)
         std::cout << msg << std::endl;
  }
}; 
static Logger gLogger;
ICudaEngine *engines[MAXDEVICENUM];
IExecutionContext *contexts[MAXDEVICENUM];
float* input_layer[MAXDEVICENUM];
float* output_layer[MAXDEVICENUM];
pthread_t threads[MAXDEVICENUM];
PthreadParams params[MAXDEVICENUM];

void caffeToGIEModel(const char *deployFile,       // name for caffe prototxt
           const char *modelFile,        // name for model
           const std::vector<std::string>& outputs,   // network outputs
           unsigned int maxBatchSize,         // batch size - NB must be at least as large as the batch we want to run with)
           int gpuidx,
           int netidx)
{
  // create the builder
//  printf("idx : %d\n",gpuidx);
  cudaSetDevice(gpuidx);
//  printf("call caffeToGIEModel\n");
  IBuilder* builder = createInferBuilder(gLogger);
//  printf("call createInferBuilder\n");

  // parse the caffe model to populate the network, then set the outputs
  INetworkDefinition* network = builder->createNetwork();
//  printf("call builder->createNetwork\n");
  ICaffeParser* parser = createCaffeParser();
//  printf("call createCaffeParser\n");
  const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile,modelFile,*network,DataType::kFLOAT);
//  printf("call parser->parse\n");

  // specify which tensors are outputs
  for (auto& s : outputs)
    network->markOutput(*blobNameToTensor->find(s.c_str()));
//  printf("call markOutput\n");
  // Build the engine
  builder->setMaxBatchSize(maxBatchSize);
//  printf("call setMaxBatchSize\n");
  builder->setMaxWorkspaceSize(1 << 30);
//  printf("call setMaxWorkspaceSize\n");

  engines[netidx]  = builder->buildCudaEngine(*network);
  assert(engines[netidx]);
//  printf("call buildCudaEngine\n");

  // we don't need the network any more, and we can destroy the parser
  network->destroy();
//  printf("call network->destroy\n");
//  parser->destroy();
//  printf("call parser->destroy\n");
  builder->destroy();
//  printf("call builder->destroy\n");
//  printf("call caffeToGIEModel done\n");
}


int Init(int *gpus_,int gpunum_,int *nets_,int netnum_,char *prototxt_,char *model_){
  printf("Init ..\n");
  int *gpus = gpus_;
  int *nets = nets_;
  int gpunum = gpunum_;
  if(gpunum > MAXDEVICENUM){
    printf("error : gpu num(%d) is greater than MAXDECIVENUM(%d)\n",gpunum,MAXDEVICENUM);
      return -1;
  }
  if(gpunum > 0){  
    int i,j,k,ret;
    for(i = 0; i < gpunum;i++){
      printf("gpu idx : %d\n",gpus[i]);
      printf("net idx : %d\n",nets[i]);
      j = gpus[i];
      k = nets[i];
      params[k].idx = i;
      params[k].gpuidx = j;
      params[k].netidx = k;
      params[k].prototxt = prototxt_;
      params[k].model = model_;
      params[k].setdata = 0;
      caffeToGIEModel(prototxt_, model_, std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, j,k);
      cudaSetDevice(j);
      contexts[k] = engines[k]->createExecutionContext();
    }
    printf("Init done\n");
    return 0;
  }else{
    printf("Set gpus > 0 for prediction in gpu mode!\n");
    return -1;
  }
}

extern "C" int Cudacvt(float *c,unsigned char *a,int w_,int h_,int idx_);

extern "C" int Cudaclip(float *c,unsigned char *a,int w_,int h_,int idx_);

void* rungpu_thread(void *arg){
  PthreadParams *p = (PthreadParams *)arg;
  cudaError_t cudaStatus;
  cudaSetDevice(p->gpuidx);
  if(p->setdata == 0){
     printf("create GPU buffers\n");
     cudaMalloc((void **)(&(input_layer[p->netidx])), p->n * p->w * p->h * sizeof(float));
     cudaMalloc((void **)(&(output_layer[p->netidx])), p->n * p->w * p->h *sizeof(float));
     printf("create GPU buffers done\n");
     p->setdata = 1;
  }
  assert(engines[p->netidx]->getNbBindings() == 2);
  void* buffers[2] = {input_layer[p->netidx],output_layer[p->netidx]};;
  int inputIndex = engines[p->netidx]->getBindingIndex(INPUT_BLOB_NAME),
      outputIndex = engines[p->netidx]->getBindingIndex(OUTPUT_BLOB_NAME);

  Cudacvt((float *)buffers[inputIndex], &(p->data[p->idx * p->step * p->w]),p->w,p->h,p->gpuidx);
  contexts[p->netidx]->execute(p->n, buffers);
  Cudaclip((float *)buffers[outputIndex],&(p->data[p->idx * p->step * p->w]),p->w,p->h,p->gpuidx);
}

int Run(int *gpus_,int gpunum_,int *nets_,int netnum_,unsigned char *data_ptr,int w_,int h_){
    if(gpunum_ > 0){
        int i,j,k,ret;
        int step_ = h_/gpunum_;
        for(i = 0;i < gpunum_ ; i++){
            j = gpus_[i];
            k = nets_[i];
            params[k].step = step_;
            params[k].n = 1;
            params[k].c = 1;
            params[k].w = w_;
            if(i == gpunum_ -1)
              params[k].h = h_ - i * step_;
            else
              params[k].h = step_;
            params[k].data = data_ptr;
            ret = pthread_create(&threads[k],NULL,rungpu_thread,&params[k]);
            if(ret !=0 ){
              printf("Create thread %d error!\n",i);
              return -1;
            }
        }
        for(i = 0; i < gpunum_;i++){
            k = nets_[i];
            pthread_join(threads[k],NULL);
            params[k].data = NULL;
        }
        return 0;
    }
}


void Uninit(int *nets_,int netnum_){
  printf("Uninit...\n");
  int i,j;
  for(i = 0;i<netnum_;i++){
//      printf("net idx : %d\n",nets_[i]);
      j = nets_[i];
      contexts[j]->destroy();
      cudaFree(input_layer[j]);
      cudaFree(output_layer[j]);
      engines[j]->destroy();
  }
   printf("Uninit done\n");
}



extern "C" int pred_init(int *gpus_,int gpunum_,int *nets_,int netnum_,char *prototxt_file_,char *model_file_);
int pred_init(int *gpus_,int gpunum_,int *nets_,int netnum_,char *prototxt_file_,char *model_file_){
    if(gpunum_ > 0){
        int ret;
        ret = Init(gpus_,gpunum_,nets_,netnum_,prototxt_file_,model_file_);
        if(ret != 0){
            printf("Init predictor error!\n");
            return -1;
        }
        return ret;
    }else{
        printf("Init gpu error : gpunum <= 0\n");
        return -1;
    }
}

extern "C" int pred_run(int *gpus_,int gpunum_,int *nets_,int netnum_,unsigned char *input,int w,int h);
int pred_run(int *gpus_,int gpunum_,int *nets_,int netnum_,unsigned char *input,int w,int h){
    int ret;
    ret = Run(gpus_,gpunum_,nets_,netnum_,input,w,h);
    return ret;
}

extern "C" void pred_uninit(int *nets_,int netnum_);
void pred_uninit(int *nets_,int netnum_){
    Uninit(nets_,netnum_);
}

