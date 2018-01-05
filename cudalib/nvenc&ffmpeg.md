
### GPU环境安装


ref : http://www.dedoimedo.com/computers/centos-7-nvidia.html

this better : http://www.tecmint.com/install-nvidia-drivers-in-linux/

```
yum update
yum install gcc kernel-devel
reboot

```
source : http://us.download.nvidia.com/XFree86/Linux-x86_64/375.39/NVIDIA-Linux-x86_64-375.39.run

```
wget http://mtb-sz-in.oss-cn-shenzhen-internal.aliyuncs.com/nvidia/NVIDIA-Linux-x86_64-375.39.run

sh NVIDIA-Linux-x86_64-375.39.run
```
### disable Nouveau driver
```
   echo "blacklist nouveau" > /etc/modprobe.d/blacklist.conf
   
   mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r).img.bak
     
   dracut -v /boot/initramfs-$(uname -r).img $(uname -r)
```
source: https://developer.nvidia.com/cuda-downloads

```
wget http://mtb-sz-in.oss-cn-shenzhen-internal.aliyuncs.com/nvidia/cuda-repo-rhel7-8-0-local-ga2-8.0.61-1.x86_64-rpm

1. sudo rpm -i cuda-repo-rhel7-8-0-local-ga2-8.0.61-1.x86_64-rpm
2. sudo yum clean all
3. sudo yum install cuda

```
http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4ZV9xsY4k

post-install action

### 安装lib

```
  cd /home/admin
    
  wget "http://mts-sh-in.vpc100-oss-cn-shanghai.aliyuncs.com/dncnn/install_gpu.sh" 
   
  chmod +x install_gpu.sh
  
  sudo su admin
  
  ./install_gpu.sh "http://mts-sh-in.vpc100-oss-cn-shanghai.aliyuncs.com/dncnn/thmodelzoo.zip" "http://mts-sh-in.vpc100-oss-cn-shanghai.aliyuncs.com/dncnn/pack.zip"
  
  exit
  
  sudo vim /etc/ld.so.conf
  
  add '/home/admin/gpu-lib/lib'
  
  sudo ldconfig
```
   
### 更新及安装cudnn

  &nbsp;**移除cudnn v5.1**
  
  ```
  rm -rf /usr/local/cuda/include/cudnn.h
  
  rm -rf /usr/local/cuda/lib64/libcudnn.so
  
  rm -rf /usr/local/cuda/lib64/libcudnn.so.5
  
  rm -rf /usr/local/cuda/lib64/libcudnn.so.5.1.10
  ```
 
  &nbsp;**安装cudnn v6.0**
  
  ```
  wget "http://mts-sh-in.vpc100-oss-cn-shanghai.aliyuncs.com/dncnn/cudnn-8.0-linux-x64-v6.0.tgz"   
  
  tar zxvf cudnn-8.0-linux-x64-v6.0.tgz 
   
  cd cuda 
  
  sudo cp lib64/lib* /usr/local/cuda/lib64/  
  
  sudo cp include/cudnn.h /usr/local/cuda/include/
  ```
  
### 更新网络连接
  
  ```
  cd /usr/local/cuda/lib64/ 
  
  sudo chmod +r libcudnn.so.6.0.20  # 自己查看.so的版本  
  
  sudo ln -sf libcudnn.so.6.0.20 libcudnn.so.6  

  sudo ln -sf libcudnn.so.6 libcudnn.so  
  
  sudo ldconfig
  ```

### 添加环境变量
  ```
  vim ~/.bashrc  
  
  export PATH=$PATH:/usr/local/cuda/bin
  
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/local/cuda/lib64   
  
  source ~/.bashrc
  
  vim /etc/ld.so.conf.d/cuda.conf
  
  /usr/local/cuda/lib64
  
  sudo ldconfig
  ```
### 更新cudnn.torch  
```     
  luarocks remove cudnn
  
  wget "http://mts-sh-in.vpc100-oss-cn-shanghai.aliyuncs.com/dncnn/cudnn.torch-R6.zip"
  
  unzip cudnn.torch-R6.zip
  
  cd cudnn.torch-R6
  
  luarocks make cudnn-scm-1.rockspec
  
  cd ../
  
  rm -rf cudnn.torch-R6.zip
  
  rm -rf cudnn.torch-R6/
```

### 加入NVENC

 ```
  unzip Video_Codec_SDK_7.1.9
  
  sudo cp Video_Codec_SDK_7.1.9/Samples/common/inc/*h /usr/local/include/
 ``` 
 
### FFmpeg

  ```
  export PKG_CONFIG_PATH=/home/admin/gpu-lib/lib/pkgconfig
  
  ./configure --enable-gpl --enable-libx265 --enable-libx264 --enable-nonfree --enable-libfdk_aac --extra-cflags=-I/home/admin/gpu-lib/include --extra-ldflags=-L/home/admin/gpu-lib/lib --enable-nvenc --enable-cuda --enable-cuvid --enable-libnpp --extra-cflags=-I/usr/local/cuda-8.0/include --extra-ldflags=-L/usr/local/cuda-8.0/lib64 --extra-ldflags=-L/usr/lib64/nvidia --extra-cflags=-I/home/admin/torch7/install/include --extra-ldflags=-L/home/admin/torch7/install/lib --extra-libs=-lTH --extra-libs=-lluaT --extra-libs=-lluajit --enable-filters
  ```
  
### FFmpeg + Caffe(on Centos 7)

#### caffe编译

&nbsp;&nbsp;参考 : "https://github.com/dex1990/caffe.git" 编译得到libcaffe.so和ffmpeg需要的libprediction.a


```     
  git clone "https://github.com/dex1990/caffe.git"

  cd caffe
    
  git checkout develop
  
  make all -j8
  
  cd cudalib
  
  ./build.sh
  
```

#### FFmpeg编译

  ```
  export PKG_CONFIG_PATH=/home/admin/gpu-lib/lib/pkgconfig
  
  ./configure  --enable-gpl --enable-libx265 --enable-libx264 --enable-nonfree --enable-libfdk_aac  --extra-cflags=-I/home/admin/gpu-lib/include --extra-ldflags=-L/home/admin/gpu-lib/lib --enable-nvenc --enable-cuda --enable-cuvid --enable-libnpp --extra-cflags=-I/usr    /local/cuda-8.0/include --extra-ldflags=-L/usr/local/cuda-8.0/lib64 --extra-ldflags=-L/usr/lib64/nvidia  --extra-libs=-lstdc++  --extra-libs='-L/mnt/6TDisk/caffe_xsy/caffe/build/lib -lcaffe -L/mnt/6TDisk/caffe_xsy/caffe/.build_release/lib -lcaffe  -lhdf5 -lhdf5_hl -    L/usr/local/cuda-8.0/lib64 -lcudart -L/usr/lib64 -lglog -lboost_system -lpthread' --extra-libs=/mnt/6TDisk/caffe_xsy/caffe/cudalib/libprediction.a --enable-filters
  
  ```
  
### FFmpeg + TensorRT(on Centos 7)

#### CUDNN v7

```
  wget "http://presigned.oss-cn-hangzhou.aliyuncs.com/sixiao/DnCNN/cudnn-8.0-linux-x64-v7.tgz"
  
  tar zxvf cudnn-8.0-linux-x64-v7.tgz
  
  sudo rm -rf /usr/local/cuda/include/cudnn.h
  
  sudo rm -rf /usr/local/cuda/lib64/libcudnn*
  
  sudo cp cuda/lib64/lib* /usr/local/cuda/lib64/
  
  sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
  
  sudo chmod +r /usr/local/cuda/lib64/libcudnn.so.7.0.4
  
  sudo ln -sf /usr/local/cuda/lib64/libcudnn.so.7.0.4 /usr/local/cuda/lib64/libcudnn.so.7
  
  sudo ln -sf /usr/local/cuda/lib64/libcudnn.so.7 /usr/local/cuda/lib64/libcudnn.so
  
  sudo ldconfig
```

#### TensorRT


```     
  wget "http://presigned.oss-cn-hangzhou.aliyuncs.com/sixiao/DnCNN/TensorRT-3.0.1.Ubuntu-14.04.5.x86_64.cuda-8.0.cudnn7.0.tar.gz"
  
  tar -zxvf TensorRT-3.0.1.Ubuntu-14.04.5.x86_64.cuda-8.0.cudnn7.0.tar.gz
  
  sudo mv TensorRT-3.0.1 /home/admin/TensorRT-3.0.1
  
  wget "http://presigned.oss-cn-hangzhou.aliyuncs.com/sixiao/DnCNN/cudalib/prediction.cpp"
  
  wget "http://presigned.oss-cn-hangzhou.aliyuncs.com/sixiao/DnCNN/cudalib/process.cu"
  
  wget "http://presigned.oss-cn-hangzhou.aliyuncs.com/sixiao/DnCNN/cudalib/build.sh"
  
  sudo ./build.sh
  
  sudo mv libprediction.a /home/admin/TensorRT-3.0.1/
  
```

#### FFmpeg编译

  ```
  export PKG_CONFIG_PATH=/home/admin/gpu-lib/lib/pkgconfig
  
  ./configure  --enable-gpl --enable-libx265 --enable-libx264 --enable-nonfree --enable-libfdk_aac  --extra-cflags=-I/home/admin/gpu-lib/include --extra-ldflags=-L/home/admin/gpu-lib/lib --enable-nvenc --enable-cuda --enable-cuvid --enable-libnpp --extra-cflags=-I/usr/local/cuda-8.0/include --extra-ldflags=-L/usr/local/cuda-8.0/lib64 --extra-ldflags=-L/usr/lib64/nvidia  --extra-libs=-lstdc++  --extra-libs='-std=c++11 -I/home/admin/TensorRT-3.0.1/include -I/usr/local/include -I/usr/local/cuda/include -L/home/admin/TensorRT-3.0.1/lib -lnvinfer -lnvparsers -lnvinfer_plugin -L/usr/local/cuda-8.0/lib64 -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart  -L/usr/lib64 -lrt -ldl -lpthread' --extra-libs=/home/admin/TensorRT-3.0.1/lib/libprediction.a --enable-filters
  
  ```







