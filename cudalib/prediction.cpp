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


struct PthreadParams{
    int idx;
    int step;
    int n;
    int c;
    int w;
    int h;
};

shared_ptr<Net<float> > *nets;
//float *data;
unsigned char *data;
pthread_t *threads;
PthreadParams *params;
int gpus;
char *prototxt;
char *model;

void* initgpu_thread(void *arg){
  PthreadParams *p = (PthreadParams *)arg;
  Caffe::SetDevice(p->idx);
  Caffe::set_mode(Caffe::GPU);
  nets[p->idx].reset(new Net<float>(prototxt,caffe::TEST));
  nets[p->idx]->CopyTrainedLayersFrom(model);
  nets[p->idx]->ClearParamDiffs();
 //  printf("init thread %d %d\n",&(threads[p->idx]),&(nets[p->idx]));
}

int Init(int gpus_,char *prototxt_,char *model_){
  printf("Init ..\n");
  ::google::InitGoogleLogging("log.txt");
  gpus = gpus_;
  prototxt = prototxt_;
  model = model_;
  if(gpus > 0){
    nets = new shared_ptr<Net<float> >[gpus];
    threads = new pthread_t[gpus];
    params = new PthreadParams[gpus];
    int i,ret;
    for(i = 0; i < gpus;i++){
/*      Caffe::SetDevice(i);
      Caffe::set_mode(Caffe::GPU);
      nets[i].reset(new Net<float>(prototxt,caffe::TEST));
      nets[i]->CopyTrainedLayersFrom(model);
      nets[i]->ClearParamDiffs();
*/  
    params[i].idx = i;
    ret = pthread_create(&threads[i],NULL,initgpu_thread,&params[i]);
    }
    for(i = 0;i < gpus;i++){
        pthread_join(threads[i],NULL);
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
  Caffe::SetDevice(p->idx);
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
  Blob<float>* input_blobs = nets[p->idx]->input_blobs()[0];
  input_blobs->Reshape(p->n,p->c,p->h,p->w);
  nets[p->idx]->Reshape();
//  start = clock();
//  cudaMemcpy(input_blobs->mutable_gpu_data(), &(data[p->idx * p->step * p->w]),
//              sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);
  Cudacvt(input_blobs->mutable_gpu_data(), &(data[p->idx * p->step * p->w]),p->w,p->h,p->idx);
//  finish = clock();
//  printf("cudaMemcpy cost time %d ms\n",(finish-start)/1000);
//  start = clock();
  nets[p->idx]->Forward();
//  finish = clock();
//  printf("Forward cost time %d ms\n",(finish-start)/1000);
//  start = clock();
  Blob<float>* output_layer = nets[p->idx]->output_blobs()[0];
  float* begin = output_layer->mutable_gpu_data();
  Cudaclip(begin,&(data[p->idx * p->step * p->w]),p->w,p->h,p->idx);
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

int Run(unsigned char *data_ptr,int w_,int h_){
    if(gpus > 0){
//        printf("Run...\n");
        int i,ret;
        int step_ = h_/gpus;
        clock_t start,finish;
        data = data_ptr;
/*        start = clock();
        data = new float [w_*h_];
        for(i = 0;i < w_*h_; i++){
            data[i] = static_cast<float>(data_ptr[i])/255.0;
        }

        finish = clock();
        printf("copy data from ffmpeg cost time %d ms\n",(finish-start)/1000);
*/
        start = clock();
        for(i = 0;i < gpus ; i++){
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
            params[i].idx = i;
            params[i].step = step_;
            params[i].n = 1;
            params[i].c = 1;
            params[i].w = w_;
            if(i == gpus -1)
              params[i].h = h_ - i * step_;
            else
              params[i].h = step_;
            ret = pthread_create(&threads[i],NULL,rungpu_thread,&params[i]);
            if(ret !=0 ){
              printf("Create thread %d error!\n",i);
              return -1;
            }
        }
        for(i = 0; i < gpus;i++){
            pthread_join(threads[i],NULL);
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
        data = NULL;
        return 0;
    }
}


void Uninit(){
  printf("Uninit...\n");
  delete []nets;
  delete []threads;
  delete []params;
  printf("Uninit done\n");
}



extern "C" int pred_init(int gpus_,char *prototxt_file_,char *model_file_);
int pred_init(int gpus_,char *prototxt_file_,char *model_file_){
    if(gpus_ > 0){
        int ret;
//        printf("%d %s %s\n",gpus_,prototxt_file_,model_file_);
        ret = Init(gpus_,prototxt_file_,model_file_);
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

extern "C" int pred_run(unsigned char *input,int w,int h);
int pred_run(unsigned char *input,int w,int h){
    int ret;
 //   clock_t start,finish;
 //   start=clock();
    ret = Run(input,w,h);
 //   finish=clock();
 //   printf("Run cost time %d ms\n",(finish-start)/1000);
    return ret;
}

extern "C" void pred_uninit();
void pred_uninit(){
    Uninit();
}

