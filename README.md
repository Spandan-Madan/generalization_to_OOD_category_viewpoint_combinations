<div align="center">
<h3>When and How CNNs generalize to out-of-distribution category-viewpoint combinations.</h3>
  <img src="docs/images/fig_1_github.png" alt="Teaser Figure">
  <a href="https://arxiv.org/abs/2007.08032">Paper</a> •
  <a href="#overview">Overview</a> •
  <a href="#findings">Findings</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#citation">Citation</a>
</div>

This repository contains the official implementation of our paper published in Nature Machine Intelligence: *When and how convolutional neural networks generalize to out-of-distribution category and viewpoint combinations*. Here you can find the code, and the newly introduced Biased-Cars dataset.

The paper can be accessed [here](https://arxiv.org/abs/2007.08032).

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

# Using the codebase

1. SETUP: Please use the `requirements.txt` file for dependencies using: `pip install requirements.txt`

2. DATASETS:

    a. Downloading: For every dataset, there is a download script under `utils`. Please use:

    ```
    cd utils
    bash download_biased_cars.sh
    ```
    b. Understanding data structure:

3. DEMOS: We provide easy demos which showcase our main experiments. These include:

    a. Impact of increasing data diversity on out-of-distribution performance: Please check [here](`https://github.com/Spandan-Madan/generalization_to_OOD_category_viewpoint_cominations/blob/main/demos/increasing_in_distribution_combinations.ipynb`)

# Findings

Our key findings are as follows:

- For a constant dataset size, increasing data diversity makes the task more challenging, as reflected in the deteriorating in-distribution perfor- mance. Yet, increasing data diversity substantially improves performance on OOD combi- nations.

- Stark contrast in in-distribution and OOD performances: Separate architectures significantly outperform shared ones on OOD combinations, eventhough they can lead to marginal drop in in-distribution performance. Thus, the belief that Shared architectures outperform Separate ones when tasks are synergistic should be revisited (Caruana, 1997), as their relative performance strongly depends on whether the test sample is in-distribution or OOD.

- Neural specialization (proposed in our paper) facilitates generalization to OOD combinations for multi-task prediction of category and viewpoint prediction. We show the emergence of two types of neurons, one driving OOD generalization for category, and the other for view- point. This corresponds to neurons selective to a category and invariant to viewpoint, and vice versa. We show that the CNN generalization behavior trends correlates with the degree of specialization of the neurons.


# Datasets

## Biased-Cars: A photo-realistic, complex dataset for OOD generalization

![Biased-Cars samples GIF](docs/images/biased_cars_samples.gif)

We introduce a challenging, photo-realistic dataset for analyzing out-of-distribution performance in computer vision: the Biased-Cars dataset. Our dataset offers complete control over the joint distribution of categories, viewpoints, and other scene parameters, and the use of physically based rendering ensures photo-realism. Some features of our dataset:

- Photorealism with diversity: Outdoor scenes with fine control over scene clutter (trees, street furniture, and pedestrians), car colors, object occlusions, diverse backgrounds (building/road textures) and lighting conditions (sky maps).

- Fine grained control: 30K images of five different car models with different car colors seen from different viewpoints car colors varying between 0-90 degrees of azimuth, and 0-50 degrees of zenith across multiple scales.

- Labels for several computer vision tasks: We provide labels for car model, color, viewpoint and scale. We also provide semantic label maps for background categories including road, sky, pavement, pedestrians, trees and buildings.

#### To download the dataset, please use:

```
cd utils
bash download_biased_cars.sh
```
Our dataset can be downloaded [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/F1NQ3R).

For ease, an example showing loading and training with the Biased-Cars dataset has been provided under tutorials.

## Previously published datasets used in our analysis
- iLab Dataset: http://ilab.usc.edu/ilab2m/iLab-2M.tar.gz
- MNIST-Rotation: https://www.dropbox.com/s/wdws3b3fjo190sk/self_generated.tar.gz?dl=0
- UIUC 3D Dataset: http://www.eecs.umich.edu/vision/data/3Ddataset.zip

## Citation

@misc{madan2021cnns,
      title={When and how CNNs generalize to out-of-distribution category-viewpoint combinations},
      author={Spandan Madan and Timothy Henry and Jamell Dozier and Helen Ho and Nishchal Bhandari and Tomotake Sasaki and Frédo Durand and Hanspeter Pfister and Xavier Boix},
      year={2021},
      eprint={2007.08032},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
