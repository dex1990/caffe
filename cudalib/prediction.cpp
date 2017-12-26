#include <pthread.h>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
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
};

class Logger : public nvinfer1::ILogger
{
  public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override
  {
    // suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
  }
};

Logger gLogger[MAXDEVICENUM];
IRuntime *runtimes[MAXDEVICENUM];
ICudaEngine *engines[MAXDEVICENUM];
IExecutionContext *contexts[MAXDEVICENUM];
pthread_t threads[MAXDEVICENUM];
PthreadParams params[MAXDEVICENUM];

void caffeToGIEModel(const char *deployFile,       // name for caffe prototxt
           const char *modelFile,        // name for model
           const std::vector<std::string>& outputs,   // network outputs
           unsigned int maxBatchSize,         // batch size - NB must be at least as large as the batch we want to run with)
           IHostMemory *&gieModelStream,      // output buffer for the GIE model
           int idx)
{
  // create the builder
  printf("call caffeToGIEModel\n");
  IBuilder* builder = createInferBuilder(gLogger[idx]);

  // parse the caffe model to populate the network, then set the outputs
  INetworkDefinition* network = builder->createNetwork();
  ICaffeParser* parser = createCaffeParser();
  const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile,modelFile,*network,DataType::kFLOAT);

  // specify which tensors are outputs
  for (auto& s : outputs)
    network->markOutput(*blobNameToTensor->find(s.c_str()));

  // Build the engine
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(1 << 20);

  ICudaEngine* engine = builder->buildCudaEngine(*network);
  assert(engine);

  // we don't need the network any more, and we can destroy the parser
  network->destroy();
  parser->destroy();

  // serialize the engine, then close everything down
  gieModelStream = engine->serialize();
  engine->destroy();
  builder->destroy();
  shutdownProtobufLibrary();
  printf("call caffeToGIEModel done\n");
}

void* initgpu_thread(void *arg){
  PthreadParams *p = (PthreadParams *)arg;
  cudaSetDevice(p->gpuidx);
  IHostMemory *gieModelStream{nullptr};
  caffeToGIEModel(p->prototxt, p->model, std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream,p->netidx);
  runtimes[p->netidx] = createInferRuntime(gLogger[p->netidx]);
  engines[p->netidx] = runtimes[p->netidx]->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
  if(gieModelStream){ 
     gieModelStream->destroy();
  }
  contexts[p->netidx] = engines[p->netidx]->createExecutionContext();
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
      ret = pthread_create(&threads[k],NULL,initgpu_thread,&params[k]);
    }
    for(i = 0;i < gpunum;i++){
        k = nets[i];
        pthread_join(threads[k],NULL);
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
  clock_t start,finish;
  cudaError_t cudaStatus;
  start = clock();
  cudaSetDevice(p->gpuidx);
  const ICudaEngine& engine = contexts[p->netidx]->getEngine();
  assert(engine.getNbBindings() == 2);
  void* buffers[2];
  int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
      outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
  cudaStatus = cudaMalloc(&buffers[inputIndex], p->n * p->h * p->w * sizeof(float));
  if (cudaStatus != cudaSuccess) {
     fprintf(stderr, "cudaMalloc failed!");
     return NULL;
  }

  cudaStatus = cudaMalloc(&buffers[outputIndex], p->n * p->h * p->w * sizeof(float));
  if (cudaStatus != cudaSuccess) {
     fprintf(stderr, "cudaMalloc failed!");
     return NULL;
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  Cudacvt((float *)buffers[inputIndex], &(p->data[p->idx * p->step * p->w]),p->w,p->h,p->gpuidx);
  contexts[p->netidx]->enqueue(p->n, buffers, stream, nullptr);
  Cudaclip((float *)buffers[outputIndex],&(p->data[p->idx * p->step * p->w]),p->w,p->h,p->gpuidx);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  cudaFree(buffers[inputIndex]);
  cudaFree(buffers[outputIndex]);
  finish = clock();
//  printf("memcpy cost time %d ms\n",(finish-start)/1000);
}

int Run(int *gpus_,int gpunum_,int *nets_,int netnum_,unsigned char *data_ptr,int w_,int h_){
    if(gpunum_ > 0){
        int i,j,k,ret;
        int step_ = h_/gpunum_;
        clock_t start,finish;
        start = clock();
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
        finish = clock();
        return 0;
    }
}


void Uninit(int *nets_,int netnum_){
  printf("Uninit...\n");
  int i,j;
  for(i = 0;i<netnum_;i++){
      printf("net idx : %d\n",nets_[i]);
      j = nets_[i];
      contexts[j]->destroy();
      engines[j]->destroy();
      runtimes[j]->destroy();
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
        printf("Init gpu error : gpus <= 0\n");
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

