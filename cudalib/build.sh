nvcc -c process.cu -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61
#gcc -c prediction.cpp -o prediction.o  -L"/usr/local/cuda-8.0/targets/x86_64-linux/lib64" -L"/usr/local/lib" -L"/usr/local/cuda-8.0/lib64" -L"/usr/local/cuda-8.0/lib64" -L"/home/admin/TensorRT-3.0.1/lib"  -L"/usr/lib64" -I/usr/local/include -I/usr/local/cuda/include -I /home/admin/TensorRT-3.0.1/include  -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread
gcc -std=c++11 -c prediction.cpp -o prediction.o  -I/home/admin/TensorRT-3.0.1/include -I/usr/local/include -I/usr/local/cuda/include  -L/home/admin/TensorRT-3.0.1/lib -lnvinfer -lnvparsers -lnvinfer_plugin  -L/usr/local/cuda-8.0/lib64 -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart  -L/usr/lib64 -lrt -ldl -lpthread
ar -r libprediction.a process.o prediction.o 

