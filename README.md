<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#on-the-robustness-of-deep-learning-image-classification-techniques">On the robustness of deep learning image classification techniques</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
# About The Project
## On the robustness of deep learning image classification techniques

<p text-align ="center">Deep learning-based techniques is currently applied in a wide range of domains. The most significant applications of AI often involve computer vision techniques such as object identification, object tracking or image classification. 

Even though DL-based techniques have achieved great success, its accuracy may never reach 100%. In many cases, it might not be a problem since a human can easily correct these mistakes. However, in some particular cases, it might become critical.

This report presents our findings on the robustness of deep learning image classification techniques:
* Explore the vulnerability of recently developed techniques
* Survey ways to train a more robust model
* Show that the models can be attacked in multiple ways
* Introduce a way to train our model against these types of
attacks.
  </p>

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
# Getting Started

Before you continue, ensure you meet the following requirements

## Prerequisites

* Python 3.7.13
* Window (Linux or Mac OS machine is not currently supported)
* Basic understanding of machine learning

## Installation

1. Get a free Anaconda app at [https://www.anaconda.com](https://www.anaconda.com) or using colab at https://colab.research.google.com
2. Install packages
  * Pytorch
     ```sh
     pip install torch==1.10.1+rocm4.2 torchvision==0.11.2+rocm4.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
     ```
  * Torchvision
     ```sh
     pip install torchvision
     ```
  * Numpy
     ```sh
     pip install numpy
     ```
  * Pathlib
     ```sh
     pip install pathlib
     ```
  * Pathlib
     ```sh
     pip install pathlib
     ``` 
  * Glob
     ```sh
     conda install -c anaconda glob2
     ```   
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE -->
# Usage
To see our work

1. Our Traffic Sign Classification model: https://github.com/nokiddig/IT5-FTW/blob/main/traffic_sign_classify.ipynb
2. Try some attack method on our model:
* FGSM: https://github.com/nokiddig/IT5-FTW/blob/main/MyModelFGSM2.py
* PGD: https://github.com/nokiddig/IT5-FTW/blob/main/MyModelPGD.py
* Data Poisoning: 
3. Using the same method on MobileNetV2:
* FGSM: https://github.com/nokiddig/IT5-FTW/blob/main/MobilenetV2FGSM.py
* PGD: https://github.com/nokiddig/IT5-FTW/blob/main/MobilenetV2PGD.py

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
# Contact

Lê Văn Sỹ - [@nokiddig](https://www.facebook.com/SyLV224)

Project Link: [https://github.com/nokiddig/IT5-FTW](https://github.com/nokiddig/IT5-FTW)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
# Acknowledgments
* [Traffic Sign Classify](https://tek4.vn/nhan-dien-bien-bao-giao-thong-bang-cnn-keras)
* [Adversarial Attacks](https://github.com/SConsul/Adversarial_Attacks)


<p align="right">(<a href="#top">back to top</a>)</p>
