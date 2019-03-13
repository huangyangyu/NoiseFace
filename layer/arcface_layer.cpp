#include <algorithm>
#include <vector>
#include <math.h>
#include "caffe/layers/arcface_layer.hpp"

namespace caffe
{

template <typename Dtype>
void ArcfaceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top)
{
  const ArcfaceParameter& param = this->layer_param_.arcface_param();
  m_ = param.m();
  sin_m = sin(m_);
  cos_m = cos(m_);
  threshold = cos(M_PI - m_);
  //transform_test_ = param.transform_test();// & (this->phase_ == TRAIN);
  
  CHECK_GE(m_, 0.0);
  CHECK_LT(m_, M_PI);
}

template <typename Dtype>
void ArcfaceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top)
{
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ArcfaceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top)
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);
  //if (!transform_test_ && this->phase_ == TEST) return;

  for (int i = 0; i < num; ++i)
  {
    int gt = static_cast<int>(label_data[i]);
    if (gt < 0) continue;
    
    Dtype cos_theta = bottom_data[i * dim + gt];
    if (cos_theta > +1.0f) cos_theta = +1.0f;
    if (cos_theta < -1.0f) cos_theta = -1.0f;
    Dtype sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    if (cos_theta >= threshold && sin_theta > 1e-6)
    {
      top_data[i * dim + gt] = cos_theta * cos_m - sin_theta * sin_m;
    }
    else
    {
      top_data[i * dim + gt] = cos_theta - sin(M_PI - m_) * m_;
      //top_data[i * dim + gt] = cos_theta;
    }
  }
}

template <typename Dtype>
void ArcfaceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom)
{
  if (top[0] != bottom[0] && propagate_down[0])
  {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    caffe_copy(count, top_diff, bottom_diff);
    //if (!transform_test_ && this->phase_ == TEST) return;

    for (int i = 0; i < num; ++i)
    {
      int gt = static_cast<int>(label_data[i]);
      if (gt < 0) continue;
    
      Dtype cos_theta = bottom_data[i * dim + gt];
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
      bottom_diff[i * dim + gt] = coffe * top_diff[i * dim + gt];
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ArcfaceLayer);
#endif

INSTANTIATE_CLASS(ArcfaceLayer);
REGISTER_LAYER_CLASS(Arcface);
}  // namespace caffe
