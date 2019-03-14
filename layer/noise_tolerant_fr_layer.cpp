// by huangyangyu, 2018/9/24
#include <algorithm>
#include <cfloat>
#include <vector>
#include <assert.h>

#include "caffe/layers/noise_tolerant_fr_layer.hpp"

namespace caffe {
  int delta(int a, int b)
  {
      if (b == -1) return a - b;
      if (a > b) return +1;
      if (a < b) return -1;
      return 0;
  }

  template <class T>
  T clamp(T x, T min, T max)
  {
      if (x > max) return max;
      if (x < min) return min;
      return x;
  }

  template <typename Dtype>
  Dtype NoiseTolerantFRLayer<Dtype>::cos2weight(const Dtype cos)
  {
      int bin_id = get_bin_id(cos);
      Dtype x, u, a;
      Dtype alpha, beta, weight1, weight2, weight3, weight;

      // focus on all samples
      // weight1: normal state [0, 1]
      // constant value
      weight1 = 1.0;

      // focus on simple&clean samples, we just offer some methods, you can try other method to archive the same purpose and get better effect.
      // someone interested in this method can try gradient-ascend-like methods, such as LRelu/tanh and so on, which we have simply tried and failed, but i still think it may make sense.
      // weight2: activative func [0, 1]
      // activation function, S or ReLu or SoftPlus
      // SoftPlus
      weight2 = clamp<Dtype>(log(1.0 + exp(10.0 * (bin_id - lt_bin_id_) / (r_bin_id_ - lt_bin_id_))) / log(1.0 + exp(10.0)), 0.0, 1.0);
      // ReLu (linear)
      //weight2 = clamp<Dtype>(1.0 * (bin_id - lt_bin_id_) / (r_bin_id_ - lt_bin_id_), 0.0, 1.0);
      // Pow Fix
      //weight2 = pow(clamp<Dtype>(1.0 * (bin_id - l_bin_id_) / (r_bin_id_ - l_bin_id_), 0.0, 1.0), 3.0);
      // S
      //weight2 = pcf_[bin_id];

      // focus on semi-hard&clean samples, we just offer some methods, you can try other method to archive the same purpose and get better effect.
      // weight3: semi-hard aug [0, 1]
      // gauss distribution
      x = bin_id;
      u = rt_bin_id_;
      //a = (r_bin_id_ - u) / r;// symmetric
      a = x > u ? ((r_bin_id_ - u) / r) : ((u - l_bin_id_) / r);// asymmetric
      weight3 = exp(-1.0 * (x - u) * (x - u) / (2 * a * a));
      // linear
      //a = (r_bin_id_ - u);// symmetric
      //a = x > u ? (r_bin_id_ - u) : (u - l_bin_id_);// asymmetric
      //weight3 = clamp<Dtype>(1.0 - fabs(x - u) / a, 0.0, 1.0);

      // without stage3
      //weight3 = weight2;

      // merge weight
      switch (func_)
      {
          case NoiseTolerantFRParameter_Func_AUTO_FIT:
              alpha = clamp<Dtype>(get_cos(r_bin_id_), 0.0, 1.0);//[0, 1]
              beta = 2.0 - 1.0 / (1.0 + exp(5-20*alpha)) - 1.0 / (1.0 + exp(20*alpha-15));//[0, 1]
              // linear
              //beta = fabs(2.0 * alpha - 1.0);//[0, 1]

              // alpha = 0.0 => beta = 1.0, weight = weight1
              // alpha = 0.5 => beta = 0.0, weight = weight2
              // alpha = 1.0 => beta = 1.0, weight = weight3
              weight = beta*(alpha<0.5) * weight1 + (1-beta) * weight2 + beta*(alpha>0.5) * weight3;//[0, 1]
              break;
          default:
              break;
      }
      // weight = 1.0;// normal method
      return weight;
  }

  template <typename Dtype>
  int NoiseTolerantFRLayer<Dtype>::get_bin_id(const Dtype cos)
  {
      int bin_id = bins_ * (cos - value_low_) / (value_high_ - value_low_);
      bin_id = clamp<int>(bin_id, 0, bins_);
      return bin_id;
  }

  template <typename Dtype>
  Dtype NoiseTolerantFRLayer<Dtype>::get_cos(const int bin_id)
  {
      Dtype cos = value_low_ + (value_high_ - value_low_) * bin_id / bins_;
      cos = clamp<Dtype>(cos, value_low_, value_high_);
      return cos;
  }

  template <typename Dtype>
  void NoiseTolerantFRLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                                    const vector<Blob<Dtype>*>& top)
  {
      const NoiseTolerantFRParameter& noise_tolerant_fr_param = this->layer_param_.noise_tolerant_fr_param();
      func_ = noise_tolerant_fr_param.func();//1
      shield_forward_ = noise_tolerant_fr_param.shield_forward();//2
      start_iter_ = noise_tolerant_fr_param.start_iter();//3
      bins_ = noise_tolerant_fr_param.bins();//4
      slide_batch_num_ = noise_tolerant_fr_param.slide_batch_num();//5
      value_low_ = noise_tolerant_fr_param.value_low();//6
      value_high_ = noise_tolerant_fr_param.value_high();//7
      debug_ = noise_tolerant_fr_param.debug();//8
      debug_prefix_ = noise_tolerant_fr_param.debug_prefix();//9

      CHECK_GE(start_iter_, 1) << "start iteration must be large than or equal to 1";
      CHECK_GE(bins_, 1) << "bins must be large than or equal to 1";
      CHECK_GE(slide_batch_num_, 1) << "slide batch num must be large than or equal to 1";
      CHECK_GT(value_high_, value_low_) << "high value must be large than low value";
      
      iter_ = 0;
      noise_ratio_ = 0.0;
      l_bin_id_ = -1;
      lt_bin_id_ = -1;
      rt_bin_id_ = -1;
      r_bin_id_ = -1;
      //start_iter_ >?= slide_batch_num_;
      start_iter_ = std::max(start_iter_, slide_batch_num_);
      pdf_ = std::vector<Dtype>(bins_+1, Dtype(0.0));
      clean_pdf_ = std::vector<Dtype>(bins_+1, Dtype(0.0));
      noise_pdf_ = std::vector<Dtype>(bins_+1, Dtype(0.0));
      pcf_ = std::vector<Dtype>(bins_+1, Dtype(0.0));
  }

  template <typename Dtype>
  void NoiseTolerantFRLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
                                                 const vector<Blob<Dtype>*>& top)
  {
      CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "The size of bottom[0] and bottom[1] don't match";
      
      if (top[0] != bottom[0]) top[0]->ReshapeLike(*bottom[0]);
      // weights
      vector<int> weights_shape(1);
      weights_shape[0] = bottom[0]->num();
      weights_.Reshape(weights_shape);
  }

  template <typename Dtype>
  void NoiseTolerantFRLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                                                     const vector<Blob<Dtype>*>& top)
  {
      skip_ = false;

      const Dtype* noise_data = NULL;
      if (bottom.size() == 4)
      {
          noise_data = bottom[3]->cpu_data();
      }
      const Dtype* label_data = bottom[2]->cpu_data();
      const Dtype* cos_data = bottom[1]->cpu_data();
      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* top_data = top[0]->mutable_cpu_data();
      Dtype* weight_data = weights_.mutable_cpu_data();

      int count = bottom[0]->count();
      int num = bottom[0]->num();// batch_size
      int dim = count / num;// c * h * w

      if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);
      
      if (this->phase_ != TRAIN) return;

      // update probability distribution/density function
      // add
      for (int i = 0; i < num; i++)
      {
          int gt = static_cast<int>(label_data[i]);
          int noise_gt = -1;
          if (noise_data)
          {
              noise_gt = static_cast<int>(noise_data[i]);
              queue_label_.push(noise_gt);
          }
          if (gt < 0)
          {
              queue_bin_id_.push(-1);
              continue;
          }
          
          int bin_id = get_bin_id(cos_data[i * dim + gt]);
          ++pdf_[bin_id];
          if (noise_data)
          {
              if (noise_gt == 0) ++clean_pdf_[bin_id];
              else ++noise_pdf_[bin_id];
          }
          queue_bin_id_.push(bin_id);
      }
      // del
      while (queue_bin_id_.size() > slide_batch_num_ * num)
      {
          int bin_id = queue_bin_id_.front();
          queue_bin_id_.pop();
          int noise_gt = -1;
          if (noise_data)
          {
              noise_gt = queue_label_.front();
              queue_label_.pop();
          }
          if (bin_id != -1)
          {
              --pdf_[bin_id];
              if (noise_data)
              {
                  if (noise_gt == 0) --clean_pdf_[bin_id];
                  else --noise_pdf_[bin_id];
              }
          }
      }

      /*
      // median filtering of the distribution
      std::vector<Dtype> filter_pdf(bins_+1, Dtype(0.0));
      for (int i = 1; i < bins_; i++)
      {
          filter_pdf[i] = pdf_[i-1] + pdf_[i] + pdf_[i+1] - \
                          std::min(std::min(pdf_[i-1], pdf_[i]), pdf_[i+1]) - \
                          std::max(std::max(pdf_[i-1], pdf_[i]), pdf_[i+1]);
      }
      */
      // mean filtering of the distribution
      Dtype sum_filter_pdf = 0.0;
      std::vector<Dtype> filter_pdf(bins_+1, Dtype(0.0));
      for (int i = fr; i <= bins_-fr; i++)
      {
          for (int j = i-fr; j <= i+fr; j++)
          {
              filter_pdf[i] += pdf_[j] / (fr+fr+1);
          }
          sum_filter_pdf += filter_pdf[i];
      }

      // update probability cumulative function
      pcf_[0] = filter_pdf[0] / sum_filter_pdf;
      for (int i = 1; i <= bins_; i++)
      {
          pcf_[i] = pcf_[i-1] + filter_pdf[i] / sum_filter_pdf;
      }

      ++iter_;
      if (iter_ < start_iter_) return;

      int l_bin_id, r_bin_id, lt_bin_id, rt_bin_id;
      // left/right end point of the distribution
      for (l_bin_id =     0; l_bin_id <= bins_ && pcf_[l_bin_id] <     0.5*s; ++l_bin_id);
      for (r_bin_id = bins_; r_bin_id >=     0 && pcf_[r_bin_id] > 1.0-0.5*s; --r_bin_id);
      // Basically does not happen
      if (l_bin_id >= r_bin_id)
      {
          //printf("Oops!\n");
          skip_ = true;
          return;
      }
      int m_bin_id_ = (l_bin_id + r_bin_id) / 2;
      // extreme points of the distribution
      int t_bin_id_ = std::distance(filter_pdf.begin(), std::max_element(filter_pdf.begin(), filter_pdf.end()));
      std::vector<int>().swap(t_bin_ids_);
      for (int i = std::max(l_bin_id, 5); i <= std::min(r_bin_id, bins_-5); i++)
      {
          if (filter_pdf[i] >= filter_pdf[i-1]   && filter_pdf[i] >= filter_pdf[i+1] && \
              filter_pdf[i]  > filter_pdf[i-2]   && filter_pdf[i]  > filter_pdf[i+2] && \
              filter_pdf[i]  > filter_pdf[i-3]+1 && filter_pdf[i]  > filter_pdf[i+3]+1 && \
              filter_pdf[i]  > filter_pdf[i-4]+2 && filter_pdf[i]  > filter_pdf[i+4]+2 && \
              filter_pdf[i]  > filter_pdf[i-5]+3 && filter_pdf[i]  > filter_pdf[i+5]+3)
          {
              t_bin_ids_.push_back(i);
              i += 5;
          }
      }
      if (t_bin_ids_.size() == 0) t_bin_ids_.push_back(t_bin_id_);
      // left/right extreme point of the distribution
      if (t_bin_id_ < m_bin_id_)
      {
          lt_bin_id = t_bin_id_;
          rt_bin_id = std::max(t_bin_ids_.back(), m_bin_id_);// fix
          //rt_bin_id = t_bin_ids_.back();// not fix
      }
      else
      {
          rt_bin_id = t_bin_id_;
          lt_bin_id = std::min(t_bin_ids_.front(), m_bin_id_);// fix
          //lt_bin_id = t_bin_ids_.front();// not fix
      }
      // directly assignment, the training process is ok, but not stable
      //l_bin_id_ = l_bin_id;
      //r_bin_id_ = r_bin_id;
      //lt_bin_id_ = lt_bin_id;
      //rt_bin_id_ = rt_bin_id;
      // by adding the unit gradient vector, the training process will be more stable
      l_bin_id_ += delta(l_bin_id, l_bin_id_);
      r_bin_id_ += delta(r_bin_id, r_bin_id_);
      lt_bin_id_ += delta(lt_bin_id, lt_bin_id_);
      rt_bin_id_ += delta(rt_bin_id, rt_bin_id_);

      // estimate the ratio of noise to clean
      // method1
      if (lt_bin_id_ < m_bin_id_)
      {
          noise_ratio_ = 2.0 * pcf_[lt_bin_id_];
      }
      else
      {
          noise_ratio_ = 0.0;
      }
      /*
      // method 2
      if (t_bin_ids_.size() >= 2)
      {
          noise_ratio_ = pcf_[lt_bin_id_] / (pcf_[lt_bin_id_] + 1.0 - pcf_[rt_bin_id_]);
      }
      else
      {
          if (t_bin_id_ < m_bin_id_ && pcf_[t_bin_id_] < 0.5)
          {
              noise_ratio_ = 2.0 * pcf_[t_bin_id_];
          }
          else if (t_bin_id_ > m_bin_id_ && pcf_[t_bin_id_] > 0.5)
          {
              noise_ratio_ = 1.0 - 2.0 * (1.0 - pcf_[t_bin_id_]);
          }
          else
          {
              noise_ratio_ = 0.0;
          }
      }
      */

      // compute weights
      for (int i = 0; i < num; i++)
      {
          int gt = static_cast<int>(label_data[i]);
          if (gt < 0) continue;
          weight_data[i] = cos2weight(cos_data[i * dim + gt]);
      }

      // print debug information
      /*
      if (debug_ && iter_ % 100 == start_iter_ % 100)
      {
          char file_name[256];
          sprintf(file_name, "/data/log/%s_%d.txt", debug_prefix_.c_str(), iter_);
          FILE *log = fopen(file_name, "w");
          
          fprintf(log, "debug iterations: %d\n", iter_);
                printf("debug iterations: %d\n", iter_);
          
          fprintf(log, "left: %.2f, left_top: %.2f, right_top: %.2f, right: %.2f, top_num: %d\n", get_cos(l_bin_id_), get_cos(lt_bin_id_), get_cos(rt_bin_id_), get_cos(r_bin_id_), t_bin_ids_.size());
                printf("left: %.2f, left_top: %.2f, right_top: %.2f, right: %.2f, top_num: %d\n", get_cos(l_bin_id_), get_cos(lt_bin_id_), get_cos(rt_bin_id_), get_cos(r_bin_id_), t_bin_ids_.size());
          
          fprintf(log, "pdf:\n");
                printf("pdf:\n");
          for (int i = 0; i <= bins_; i++)
          {
              fprintf(log, "%.2f %.2f\n", get_cos(i), pdf_[i]);
                    printf("%.2f %.2f\n", get_cos(i), pdf_[i]);
          }
          
          fprintf(log, "clean pdf:\n");
                printf("clean pdf:\n");
          for (int i = 0; i <= bins_; i++)
          {
              fprintf(log, "%.2f %.2f\n", get_cos(i), clean_pdf_[i]);
                    printf("%.2f %.2f\n", get_cos(i), clean_pdf_[i]);
          }

          fprintf(log, "noise pdf:\n");
                printf("noise pdf:\n");
          for (int i = 0; i <= bins_; i++)
          {
              fprintf(log, "%.2f %.2f\n", get_cos(i), noise_pdf_[i]);
                    printf("%.2f %.2f\n", get_cos(i), noise_pdf_[i]);
          }

          fprintf(log, "pcf:\n");
                printf("pcf:\n");
          for (int i = 0; i <= bins_; i++)
          {
              fprintf(log, "%.2f %.2f\n", get_cos(i), pcf_[i]);
                    printf("%.2f %.2f\n", get_cos(i), pcf_[i]);
          }
          
          fprintf(log, "weight:\n");
                printf("weight:\n");
          for (int i = 0; i <= bins_; i++)
          {
              fprintf(log, "%.2f %.2f\n", get_cos(i), cos2weight(get_cos(i)));
                    printf("%.2f %.2f\n", get_cos(i), cos2weight(get_cos(i)));
          }
          
          fprintf(log, "noise ratio: %.2f\n", noise_ratio_);
                printf("noise ratio: %.2f\n", noise_ratio_);
          
          fclose(log);
      }
      */
      
      // forward
      if (shield_forward_) return;
      
      for (int i = 0; i < num; i++)
      {
          int gt = static_cast<int>(label_data[i]);
          if (gt < 0) continue;
          for (int j = 0; j < dim; j++)
          {
              top_data[i * dim + j] *= weight_data[i];
          }
      }
  }

  template <typename Dtype>
  void NoiseTolerantFRLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                      const vector<bool>& propagate_down, 
                                                      const vector<Blob<Dtype>*>& bottom)
  {
      if (propagate_down[0])
      {
          const Dtype* label_data = bottom[2]->cpu_data();
          const Dtype* top_diff = top[0]->cpu_diff();
          Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
          const Dtype* weight_data = weights_.cpu_data();

          int count = bottom[0]->count();
          int num = bottom[0]->num();
          int dim = count / num;

          if (top[0] != bottom[0]) caffe_copy(count, top_diff, bottom_diff);

          if (this->phase_ != TRAIN) return;

          if (iter_ < start_iter_) return;

          // backward
          for (int i = 0; i < num; i++)
          {
              int gt = static_cast<int>(label_data[i]);
              if (gt < 0) continue;
              for (int j = 0; j < dim; j++)
              {
                  bottom_diff[i * dim + j] *= skip_ ? Dtype(0.0) : weight_data[i];
              }
          }
      }
  }

#ifdef CPU_ONLY
  STUB_GPU(NoiseTolerantFRLayer);
#endif

INSTANTIATE_CLASS(NoiseTolerantFRLayer);
REGISTER_LAYER_CLASS(NoiseTolerantFR);
}  // namespace caffe
