#include "caffe/caffe.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <pthread.h>
#include<ctime>

using namespace caffe;
using namespace std;

#define MAXDEVICENUM 16

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

//shared_ptr<Net<float> > *nets;
//float *data;
//pthread_t *threads;
//PthreadParams *params;
//char *prototxt;
//char *model;
shared_ptr<Net<float> > nets[MAXDEVICENUM];
pthread_t threads[MAXDEVICENUM];
PthreadParams params[MAXDEVICENUM];

void* initgpu_thread(void *arg){
  PthreadParams *p = (PthreadParams *)arg;
  Caffe::SetDevice(p->gpuidx);
  Caffe::set_mode(Caffe::GPU);
  nets[p->netidx].reset(new Net<float>(p->prototxt,caffe::TEST));
  nets[p->netidx]->CopyTrainedLayersFrom(p->model);
  nets[p->netidx]->ClearParamDiffs();
}

int Init(int *gpus_,int gpunum_,int *nets_,int netnum_,char *prototxt_,char *model_){
  printf("Init ..\n");
//  ::google::InitGoogleLogging("log.txt");
  int *gpus = gpus_;
  int *nets = nets_;
  int gpunum = gpunum_;
  if(gpunum > MAXDEVICENUM){
      printf("error : gpu num(%d) is greater than MAXDECIVENUM(%d)\n",gpunum,MAXDEVICENUM);
      return -1;
  }
//  prototxt = prototxt_;
//  model = model_;
  if(gpunum > 0){  
   // nets = new;
   // return -1; shared_ptr<Net<float> >[gpus];
   // threads = new pthread_t[gpunum];
   // params = new PthreadParams[gpunum];
    int i,j,k,ret;
    for(i = 0; i < gpunum;i++){
/*      Caffe::SetDevice(i);
      Caffe::set_mode(Caffe::GPU);
      nets[i].reset(new Net<float>(prototxt,caffe::TEST));
      nets[i]->CopyTrainedLayersFrom(model);
      nets[i]->ClearParamDiffs();
*/  
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
//  clock_t start,finish;
//  start = clock();
  Caffe::SetDevice(p->gpuidx);
  Caffe::set_mode(Caffe::GPU);
//  finish = clock();
//  printf("SetDevice cost time %d ms\n",(finish-start)/1000);
/*  float *pdata = new float[p->w * p->h];
  int i;
  for(i = 0; i < p->w * p->h; i++){
     pdata[i] = data[p->idx * p->step * p->w +i];
  }
  printf("Thread %d input %f %f %f %f\n",p->idx,pdata[0], pdata[1],pdata[2],pdata[3]);
*/
  Blob<float>* input_blobs = nets[p->netidx]->input_blobs()[0];
  input_blobs->Reshape(p->n,p->c,p->h,p->w);
  nets[p->netidx]->Reshape();
//  start = clock();
//  cudaMemcpy(input_blobs->mutable_gpu_data(), &(data[p->idx * p->step * p->w]),
//              sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);
  Cudacvt(input_blobs->mutable_gpu_data(), &(p->data[p->idx * p->step * p->w]),p->w,p->h,p->gpuidx);
//  finish = clock();
//  printf("cudaMemcpy cost time %d ms\n",(finish-start)/1000);
//  start = clock();
  nets[p->netidx]->Forward();
//  finish = clock();
//  printf("Forward cost time %d ms\n",(finish-start)/1000);
//  start = clock();
  Blob<float>* output_layer = nets[p->netidx]->output_blobs()[0];
  float* begin = output_layer->mutable_gpu_data();
  Cudaclip(begin,&(p->data[p->idx * p->step * p->w]),p->w,p->h,p->gpuidx);
//  const float* begin = output_layer->cpu_data();
/*  printf("thread %d %d %d %d %d\n",p->idx,output_layer->num(),output_layer->channels(),output_layer->height(),output_layer->width());
  for(i = 0; i < p->w * p->h; i++){
      data[p->idx * p->step * p->w + i] = begin[i];
  }
  printf("Thread %d error %f %f %f %f\n",p->idx,begin[0], begin[1],begin[2],begin[3]);
  printf("Thread %d error %f %f %f %f\n",p->idx,output_layer->data_at(0,0,0,0), output_layer->data_at(0,0,0,1),output_layer->data_at(0,0,0,2),output_layer->data_at(0,0,0,3));
  delete []pdata;*/
//  memcpy(&(data[p->idx * p->step * p->w]),&(begin[0]),sizeof(float) * p->w * p->h);
//  finish = clock();
//  printf("memcpy cost time %d ms\n",(finish-start)/1000);
}

int Run(int *gpus_,int gpunum_,int *nets_,int netnum_,unsigned char *data_ptr,int w_,int h_){
    if(gpunum_ > 0){
        int i,j,k,ret;
        int step_ = h_/gpunum_;
        clock_t start,finish;
//        data = data_ptr;
        start = clock();
        for(i = 0;i < gpunum_ ; i++){
  /*      Caffe::SetDevice(1);
        Caffe::set_mode(Caffe::GPU);
        Blob<float>* input_blobs = nets[i]->input_blobs()[0];
        input_blobs->Reshape(1,1,h_,w_);
        nets[i]->Reshape();
        cudaMemcpy(input_blobs->mutable_gpu_data(), data,
                    sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);
        for(int j =0 ;j<1;j++){
            start = clock();
            nets[i]->Forward();
            finish = clock();
            printf("Forward cost time %d ms\n",(finish-start)/1000);
        }
        Blob<float>* output_layer = nets[i]->output_blobs()[0];
        const float* begin = output_layer->cpu_data();
           for(i = 0; i < w_ * h_; i++){
            data[i] = begin[i];
       }*/
//            params[i].idx = i;
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
//        printf("run thread cost time %d ms\n",(finish-start)/1000);

//        start= clock();
/*        float res_f;
  
        for(i = 0;i < w_*h_; i++){
            res_f = (float)data_ptr[i] - data[i]*255.0;
            if(res_f < 0)
                res_f = 0;
            if(res_f > 255.0)
                res_f = 255.0;
            data_ptr[i] = (unsigned char)res_f;
        }
  
        delete []data;
        finish = clock();
        printf("copy data to ffmpeg cost time %d ms\n",(finish-start)/1000);
*/
//        data = NULL;
        return 0;
    }
}


void Uninit(int *nets_,int netnum_){
  printf("Uninit...\n");
  int i,j;
  for(i = 0;i<netnum_;i++){
      printf("net idx : %d\n",nets_[i]);
      j = nets_[i];
      memset(&(nets[j]),0,sizeof(shared_ptr<Net<float> >));
  }
//  memset(nets,0,MAXDEVICENUM*sizeof(shared_ptr<Net<float> >));
//  memset(threads,0,MAXDEVICENUM*sizeof(pthread_t));
//  memset(params,0,MAXDEVICENUM*sizeof(PthreadParams));
//  delete []threads;
//  delete []params;
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

