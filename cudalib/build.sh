nvcc -c process.cu -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61
gcc -c prediction.cpp -o prediction.o  -I/mnt/6TDisk/caffe_xsy/caffe/include -I/usr/local/include -I/usr/local/cuda/include  -L/mnt/6TDisk/caffe_xsy/caffe/build/lib -lcaffe -lhdf5 -L/usr/local/cuda-8.0/lib64 -lcudart -L/usr/lib64 -lglog -lboost_system -lpthread
ar -r libprediction.a process.o prediction.o 

