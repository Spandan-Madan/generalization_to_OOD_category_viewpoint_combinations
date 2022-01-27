# Overview
This repository contains the official implementation of our paper published in Nature Machine Intelligence: *When and how convolutional neural networks generalize to out-of-distribution category and viewpoint combinations". Here you can find the code, and the newly introduced Biased-Cars dataset.

The paper can be accessed [here](https://arxiv.org/abs/2007.08032).

Recent works suggest that convolutional neural networks (CNNs) fail to generalize to out-of-distribution (OOD) category-viewpoint combinations, ie. combinations not seen during training. In this paper, we investigate when and how such OOD generalization may be possible, and identifying the neural mechanisms that facilitate such OOD generalization.

We show that increasing the number of in-distribution combinations (ie. data diversity) substantially improves generalization to OOD combinations, even with the same amount of training data. We compare learning category and viewpoint in separate and shared network architectures, and observe starkly different trends on in-distribution and OOD combinations, ie. while shared networks are helpful in-distribution, separate networks significantly outperform shared ones at OOD combinations. Finally, we demonstrate that such OOD generalization is facilitated by the neural mechanism of specialization, ie. the emergence of two types of neurons -- neurons selective to category and invariant to viewpoint, and vice versa.


# Datasets

- Biased-Cars: https://drive.google.com/file/d/10cKaEYCPvt3pltK8T7fFhmHzRRVo4rW7/view?usp=sharing
- iLab Dataset: http://ilab.usc.edu/ilab2m/iLab-2M.tar.gz
- MNIST-Rotation: https://www.dropbox.com/s/wdws3b3fjo190sk/self_generated.tar.gz?dl=0
- UIUC 3D Dataset: http://www.eecs.umich.edu/vision/data/3Ddataset.zip
