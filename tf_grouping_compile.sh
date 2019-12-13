#/bin/bash
CUDA_ROOT=/usr/local/cuda-9.0
TF_ROOT=/home/louise/anaconda2/envs/env2/lib/python2.7/site-packages/tensorflow/
#/home/louise/anaconda2/envs/env3/lib/python2.7/site-packages/tensorflow_core/include/tensorflow/core/framework

${CUDA_ROOT}/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

echo $CUDA_ROOT
echo $TF_INC
echo $TF_LIB

# TF1.4

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I$TF_INC/ -I$TF_INC/external/nsync/public -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/tensorflow/core/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_ROOT} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

#/usr/local/cuda-8.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /home/liuyuex/anaconda2/envs/tensorflow/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework
