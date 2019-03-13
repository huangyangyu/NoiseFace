# Noise-Tolerant Paradigm for Training Face Recognition CNNs

Paper link: https://arxiv.org/abs/???

### Contents
1. [Requirements](#requirements)
1. [Dataset](#dataset)
1. [Diagram](#diagram)
1. [Performance](#performance)
1. [Contact](#contact)
1. [Citation](#citation)
1. [License](#license)

### Requirements

1. [**`Caffe`**](http://caffe.berkeleyvision.org/installation.html)

### Dataset

Training dataset:
1. [**`CASIA-Webface clean`**](https://github.com/happynear/FaceVerification)
2. [**`IMDB-Face`**](https://github.com/fwang91/IMDb-Face)
3. [**`MS-Celeb-1M`**](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)

Testing dataset:
1. [**`LFW`**](http://vis-www.cs.umass.edu/lfw/)
2. [**`AgeDB`**](https://ibug.doc.ic.ac.uk/resources/agedb/)
3. [**`CFP`**](http://www.cfpw.io/)
4. [**`MegaFace`**](http://megaface.cs.washington.edu/)

Both the training data and testing data are aligned by the method described in [**`SeqFace project`**](https://github.com/huangyangyu/SeqFace)

### Diagram

The figure shows three strategies in different purposes. At the beginning of the training process, we focus on all samples; then we focus on easy/clean samples; at last we focus on semi-hard clean samples.

![The strategy](https://raw.githubusercontent.com/huangyangyu/NoiseFace/master/figures/strategy.png)
  

The figure explains the fusion function of three strategies. The left part demonstrates three functions: **&alpha;(&delta;<sub>r</sub>)**, **&beta;(&delta;<sub>r</sub>)**, and **&gamma;(&delta;<sub>r)**. The right part shows two fusion examples. According to the **&omega;**, we can see that the easy/clean samples are emphasized in the first example(**&delta;<sub>r</sub>** < 0.5), and the semi-hard clean samples are emphasized in the second example(**&delta;<sub>r</sub>** > 0.5).
**For more detail, please click the figure to play demo video.**
[![The demo video](https://raw.githubusercontent.com/huangyangyu/NoiseFace/master/figures/detail.png)](https://youtu.be/FxRoN_i7FLw)
  

The figure shows the 2D Hist<sub>all</sub> of CNN<sub>common</sub> (up) and CNN<sub>m2</sub> (down) under 40% noise rate.
![The 2D Hist](https://raw.githubusercontent.com/huangyangyu/NoiseFace/master/figures/webface_dist_2D_noise-40.png)
  

The figure shows the 3D Hist<sub>all</sub> of CNN<sub>common</sub> (left) and CNN<sub>m2</sub> (right) under 40% noise rate.
![The 3D Hist](https://raw.githubusercontent.com/huangyangyu/NoiseFace/master/figures/webface_dist_3D_noise-40.png)

### Performance

The table shows comparison of accuracies(%) on **LFW**, **ResNet-20** models are used. CNN<sub>clean</sub> is trained with clean
data WebFace-Clean-Sub by using the traditional method. CNN<sub>common</sub> is trained with noisy dataset WebFace-All by using the traditional method. CNN<sub>ct</sub> is trained
with noisy dataset WebFace-All by using our implemented Co-teaching(with pre-given noise rates). CNN<sub>m1</sub> and CNN<sub>m2</sub> are all trained with noisy dataset WebFace-All but through the proposed approach, and they respectively use the 1st and 2nd method to compute loss.
Note: The WebFace-Clean-Sub is the clean part of the WebFace-All, the WebFace-All contains noise data with different rate as describe below.

| Loss      | Actual Noise Rate | CNN<sub>clean</sub> | CNN<sub>common</sub> | CNN<sub>ct</sub> | CNN<sub>m1</sub> | CNN<sub>m2</sub> | Estimated Noise Rate |
| --------- | ----------------- | ------------------- | -------------------- | ---------------- | ---------------- | ---------------- | -------------------- |
| L2softmax | 0%                | 94.65               | 94.65                | -                | 95.00            | **96.28**        | 2%                   |
| L2softmax | 20%               | 94.18               | 89.05                | 92.12            | 92.95            | **95.26**        | 18%                  |
| L2softmax | 40%               | 92.71               | 85.63                | 87.10            | 89.91            | **93.90**        | 42%                  |
| L2softmax | 60%               | **91.15**           | 76.61                | 83.66            | 86.11            | 87.61            | 56%                  |
| Arcface   | 0%                | 97.95               | 97.95                | -                | 97.11            | **98.11**        | 2%                   |
| Arcface   | 20%               | **97.80**           | 96.48                | 96.53            | 96.83            | 97.76            | 18%                  |
| Arcface   | 40%               | 96.53               | 92.33                | 94.25            | 95.88            | **97.23**        | 36%                  |
| Arcface   | 60%               | 94.56               | 84.05                | 90.36            | 93.66            | **95.15**        | 54%                  |


### Contact

  [Wei Hu](mailto:huwei@mail.buct.edu.cn)

  [Yangyu Huang](mailto:yangyu.huang.1990@outlook.com)


### Citation

If you find this work useful in your research, please cite  
```text
@inproceedings{Hu2019NoiseFace,
  title = {Noise-Tolerant Paradigm for Training Face Recognition CNNs},
  author = {Hu, Wei and Huang, Yangyu and Zhang, Fan and Li, Ruirui},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  month = {June},
  year = {2019},
  address = {Long Beach, CA}
}
```


### License

    The project is released under the MIT License
