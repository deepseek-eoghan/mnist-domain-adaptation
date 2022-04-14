<div align="center">

# MNISTM Domain Adaptation 

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

Domain Adaptation using the pytorch-adapt framework <br>

<a href="https://github.com/KevinMusgrave/pytorch-adapt"></a>

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<br>


### Understanding domain adaptation

Domain adaptation is a field of computer vision, where our goal is to train a neural network on a source dataset and secure a good accuracy on the target dataset which is significantly different from the source dataset. 

There are 3 types of domain adaptation based on target domain

1. Supervised
2. Semi-supervised
3. Unsupervised

There are 3 techniques for realizing domain adaptation algorithms

1. Divergence based
2. Adversarial based
3. Reconstruction based

#### Divergence

- works on the principle of minimizing some divergence-based criterion between source and target distribution, hence leading to domain invariant features.
- Contrastive Domain Discrepancy, Correlation Alignment, Maximum Mean Discrepancy (MMD), Wasserstein etc.


#### Adversarial

- generator is simply the feature extractor and we add new discriminator networks which learn to distinguish between source and target domain features. 
- discriminator helps the generator to produce features that are indistinguishable for source and target domain
- There are two losses, classification loss and discriminator loss
- Discriminator loss helps discriminator correctly classify between source and target domain features
- GRL block is a simple block that multiplies the gradient with -1 or a negative value while back-propagating


#### Reconstruction

- works on the idea of Image-to-Image translation
- learn the translation from the target domain images to source domain image and train a classifier on the source domain
- 