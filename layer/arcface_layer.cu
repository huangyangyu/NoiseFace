#include <algorithm>
#include <vector>
#include <math.h>
#include "caffe/layers/arcface_layer.hpp"

namespace caffe
{

template <typename Dtype>
__global__ void ArcfaceForward(const int n, const int dim, const Dtype* label,
                               const Dtype* bottom_data, Dtype* top_data,
                               Dtype m, Dtype cos_m, Dtype sin_m, Dtype threshold)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int gt = static_cast<int>(label[index]);
    if (gt >= 0)
    {
      Dtype cos_theta = bottom_data[index * dim + gt];
      if (cos_theta > +1.0f) cos_theta = +1.0f;
      if (cos_theta < -1.0f) cos_theta = -1.0f;
      Dtype sin_theta = sqrt(1.0f - cos_theta * cos_theta);

      if (cos_theta >= threshold && sin_theta > 1e-6)
      {
        top_data[index * dim + gt] = cos_theta * cos_m - sin_theta * sin_m;
      }
      else
      {
        top_data[index * dim + gt] = cos_theta - sin(M_PI - m) * m;
        //top_data[index * dim + gt] = cos_theta;
      }
    }
  }
}

template <typename Dtype>
__global__ void ArcfaceBackward(const int n, const int dim, const Dtype* label,
                                const Dtype* bottom_data, Dtype* bottom_diff, const Dtype* top_diff,
                                Dtype cos_m, Dtype sin_m, Dtype threshold)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int gt = static_cast<int>(label[index]);
    if (gt >= 0)
    {
      Dtype cos_theta = bottom_data[index * dim + gt];
      if (cos_theta > +1.0f) cos_theta = +1.0f;
      if (cos_theta < -1.0f) cos_theta = -1.0f;
      Dtype sin_theta = sqrt(1.0f - cos_theta * cos_theta);
      
      Dtype coffe = 0.0f;
      if (cos_theta >= threshold && sin_theta > 1e-6)
      {
        //coffe = sin(theta + m_) / sin_theta;
        coffe = cos_m + sin_m * cos_theta / sin_theta;
      }
      else
      {
        coffe = 1.0f;
      }
      bottom_diff[index * dim + gt] = coffe * top_diff[index * dim + gt];
    }
  }
}

template <typename Dtype>
void ArcfaceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top)
{
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);
  //if (!transform_test_ && this->phase_ == TEST) return;

  ArcfaceForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
    num, dim, label_data, bottom_data, top_data, m_, cos_m, sin_m, threshold);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ArcfaceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom)
{
  if (top[0] != bottom[0] && propagate_down[0])
  {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    caffe_copy(count, top_diff, bottom_diff);
    //if (!transform_test_ && this->phase_ == TEST) return;

    ArcfaceBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, label_data, bottom_data, bottom_diff, top_diff, cos_m, sin_m, threshold);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ArcfaceLayer);
}  // namespace caffe
