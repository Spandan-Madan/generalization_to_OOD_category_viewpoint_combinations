<div align="center">
<h3>When and How CNNs generalize to out-of-distribution category-viewpoint combinations.</h3>
  <img src="docs/images/fig_1_github.png" alt="Teaser Figure">
  <a href="https://arxiv.org/abs/2007.08032">Paper</a> •
  <a href="#overview">Overview</a> •
  <a href="#codebase">Using the Codebase</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#citation">Citation</a>
</div>

This repository contains the official implementation of our paper published in Nature Machine Intelligence: *When and how convolutional neural networks generalize to out-of-distribution category and viewpoint combinations*. Here you can find the code, and the newly introduced Biased-Cars dataset.

The paper can be accessed [here](https://www.nature.com/articles/s42256-021-00437-5).

<div align="center">
<h3>Authors</h3>
  <!-- <img src="docs/images/fig_1_github.png" alt="Teaser Figure"> -->
  <a href="http://people.fas.harvard.edu/~spm253/spandan/">Spandan Madan</a> •
  <a href="https://cbmm.mit.edu/about/people/henry">Timothy Henry</a> •
  <a href="https://cbmm.mit.edu/about/people/dozier">Jamell Dozier</a> •
  <a href="https://superurop.mit.edu/scholars/helen-ho/">Helen Ho</a> •
  <a href="https://www.linkedin.com/in/nishchalb/">Nishchal Bhandari</a> •
  <a href="https://cbmm.mit.edu/about/people/sasaki">Tomotake Sasaki</a> •
  <a href="https://people.csail.mit.edu/fredo/">Frédo Durand</a> •
  <a href="https://vcg.seas.harvard.edu/people/hanspeter-pfister">Hanspeter Pfister</a> •
  <a href="https://web.mit.edu/xboix/www/index.html">Xavier Boix</a>
</div>

# Overview

Recent works suggest that convolutional neural networks (CNNs) fail to generalize to out-of-distribution (OOD) category-viewpoint combinations, ie. combinations not seen during training. In this paper, we investigate when and how such OOD generalization may be possible, and identifying the neural mechanisms that facilitate such OOD generalization.

We show that increasing the number of in-distribution combinations (ie. data diversity) substantially improves generalization to OOD combinations, even with the same amount of training data. We compare learning category and viewpoint in separate and shared network architectures, and observe starkly different trends on in-distribution and OOD combinations, ie. while shared networks are helpful in-distribution, separate networks significantly outperform shared ones at OOD combinations. Finally, we demonstrate that such OOD generalization is facilitated by the neural mechanism of specialization, ie. the emergence of two types of neurons -- neurons selective to category and invariant to viewpoint, and vice versa.

# Codebase

1. SETUP: Please use the `requirements.txt` file for dependencies using: `pip install requirements.txt`

2. DATASETS:

    a. Downloading: For every dataset, there is a download script under `utils`. Bised-Cars can be downloaded using:

    ```
    cd utils
    bash download_biased_cars.sh
    ```
    For other datasets, please use other scripts in the `utils` directory.

    b. Understanding data structure: Data-splits with different number of in-distribution combinations are stored in file lists under `dataset_lists`. Names of train, test, val files describes the number of in-distribution combinations. For instance, setting `dataset_name = rotation_model_15_compositions_seen` would start the experiment for 15/25 in-distribution combinations for the Biased-cars dataset. Similarly, setting it to `ilab_comps_12_seen_shuffled_1` would start the experiment for iLab dataset with 12/36 in-distribution combinations.

3. DEMOS: We provide easy demos which showcase our main experiments. These include:

    a. Impact of increasing data diversity on out-of-distribution (OOD) performance: [LINK](https://github.com/Spandan-Madan/generalization_to_OOD_category_viewpoint_cominations/blob/main/demos/increasing_in_distribution_combinations.ipynb)

    b. Performance of `Separate` vs `Shared` architectures on in-distribution and OOD performance: [LINK](https://github.com/Spandan-Madan/generalization_to_OOD_category_viewpoint_cominations/blob/main/demos/separate_vs_shared.ipynb)

    c. Analyzing role of Neural Specialization in facilitating generalization: [LINK](https://github.com/Spandan-Madan/generalization_to_OOD_category_viewpoint_cominations/blob/main/demos/neural_activity.ipynb)

    d. Loading, training and testing on biased cars dataset: [LINK](https://github.com/Spandan-Madan/generalization_to_OOD_category_viewpoint_cominations/blob/main/demos/using_biased_cars.ipynb)

    e. Loading semantic segmentation maps for biased cars dataset (not used in the paper, but provided): [LINK](https://github.com/Spandan-Madan/generalization_to_OOD_category_viewpoint_cominations/blob/main/demos/biased_cars_semantic_segmentation.ipynb)

4. TRAINING:

    `train.py` is the entry point to allow training across iLab, Biased-Cars and MNIST-Rotation datasets. Below is an example to run the `Shared` architecture on the Biased-Cars dataset with 60% in-distribution combinations.

    ```
    python train.py --dataset_name rotation_model_15_compositions_seen --num_epochs 5 --batch_size 50 --arch LATE_BRANCHING_COMBINED --save_file_suffix test_run --task rotation --experiment_out_name biased_cars_test
    ```

    As described in the point 2 above, --dataset_name can be changed to train/test models with different number of in-distribution combinations for different datasets.

# Datasets

## Biased-Cars: A photo-realistic, complex dataset for OOD generalization

![Biased-Cars samples GIF](docs/images/biased_cars_samples.gif)

We introduce a challenging, photo-realistic dataset for analyzing out-of-distribution performance in computer vision: the Biased-Cars dataset. Our dataset offers complete control over the joint distribution of categories, viewpoints, and other scene parameters, and the use of physically based rendering ensures photo-realism. Some features of our dataset:

- Photorealism with diversity: Outdoor scenes with fine control over scene clutter (trees, street furniture, and pedestrians), car colors, object occlusions, diverse backgrounds (building/road textures) and lighting conditions (sky maps).

- Fine grained control: 30K images of five different car models with different car colors seen from different viewpoints car colors varying between 0-90 degrees of azimuth, and 0-50 degrees of zenith across multiple scales.

- Labels for several computer vision tasks: We provide labels for car model, color, viewpoint and scale. We also provide semantic label maps for background categories including road, sky, pavement, pedestrians, trees and buildings.

#### To download the dataset:

```
cd utils
bash download_biased_cars.sh
```
Our dataset can also be accessed [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/F1NQ3R).

#### To load, train and test with the dataset:

- We provide easy data loaders present under `res/loader/`.

- For sample of training and testing, please check `demos/using_biased_cars.ipynb`

## Previously published datasets used in our analysis
Our paper also provides results with the following public datasets:
- iLab Dataset: http://ilab.usc.edu/ilab2m/iLab-2M.tar.gz
- MNIST-Rotation: https://www.dropbox.com/s/wdws3b3fjo190sk/self_generated.tar.gz?dl=0
- UIUC 3D Dataset: http://www.eecs.umich.edu/vision/data/3Ddataset.zip

## Citation

To cite our paper, please use the following bibTex:

```
@Article{Madan2022,
author={Madan, Spandan
and Henry, Timothy
and Dozier, Jamell
and Ho, Helen
and Bhandari, Nishchal
and Sasaki, Tomotake
and Durand, Fr{\'e}do
and Pfister, Hanspeter
and Boix, Xavier},
title={When and how convolutional neural networks generalize to out-of-distribution category--viewpoint combinations},
journal={Nature Machine Intelligence},
year={2022},
month={Feb},
day={01},
volume={4},
number={2},
pages={146-153},
abstract={Object recognition and viewpoint estimation lie at the heart of visual understanding. Recent studies have suggested that convolutional neural networks (CNNs) fail to generalize to out-of-distribution (OOD) category--viewpoint combinations, that is, combinations not seen during training. Here we investigate when and how such OOD generalization may be possible by evaluating CNNs trained to classify both object category and three-dimensional viewpoint on OOD combinations, and identifying the neural mechanisms that facilitate such OOD generalization. We show that increasing the number of in-distribution combinations (data diversity) substantially improves generalization to OOD combinations, even with the same amount of training data. We compare learning category and viewpoint in separate and shared network architectures, and observe starkly different trends on in-distribution and OOD combinations, that is, while shared networks are helpful in distribution, separate networks significantly outperform shared ones at OOD combinations. Finally, we demonstrate that such OOD generalization is facilitated by the neural mechanism of specialization, that is, the emergence of two types of neuron---neurons selective to category and invariant to viewpoint, and vice versa.},
issn={2522-5839},
doi={10.1038/s42256-021-00437-5},
url={https://doi.org/10.1038/s42256-021-00437-5}
}
```
