# learn-from-unlearnable
Code for the paper [What Can We Learn from Unlearnable Datasets?](http://arxiv.org/abs/2206.03693) by Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein. (Accepted to NeurIPS 2023)

<div align="center">
  <img width="95%" alt="Teaser" src="imgs/repo_teaser.png">
</div>


Our Orthogonal Projection method can recover class-wise perturbations and remove them to allow for learning from poisoned data. <b>(Top)</b> A batch of 10 poisoned images from [LSP Poison](https://dl.acm.org/doi/10.1145/3534678.3539241) and [OPS Poison](https://openreview.net/forum?id=p7G8t5FVn2h). <b>(Middle)</b> Recovered perturbations after training a logistic regression model compared to computing a class-wise average image. <b>(Bottom)</b> After projecting the poisoned data, a ResNet-18 can train and achieve high test accuracy.


## Setup instructions
Create a Conda environment and install necessary dependencies:
```
conda create -n learn-unlearnable python=3.9
conda activate learn-unlearnable
pip install -r requirements.txt
```

Next, specify data paths in `constants.py`. You don't need to specify all paths, just the ones you need. For example, if you will be using a CIFAR-10 [Adversarial Poison](https://arxiv.org/abs/2106.10807), specify `CIFAR10_ROOT` and `DATA_SETUPS['error-max']['root']` (the location of the poisoned data). 

**Download Poisons**: Original repos allow you to easily download or create poisons. We provide a short list here:
- [Adversarial Poisoning](https://github.com/lhfowl/adversarial_poisons)
- [Unlearnable Examples](https://github.com/HanxunH/Unlearnable-Examples)
- [Autoregressive Perturbations](https://github.com/psandovalsegura/autoregressive-poisoning)
- [LSP](https://github.dev/dayu11/Availability-Attacks-Create-Shortcuts)
- [OPS](https://github.com/cychomatica/One-Pixel-Shotcut)

## Reproduce Results
#### Section 4.2: DNNs can learn useful features from unlearnable datasets

The following command trains a ResNet-18 on [Adversarial Poison](https://arxiv.org/abs/2106.10807) for 60 epochs:
```
python dfr_step_1.py error-max
```
The checkpoints are placed in the `model-ckpts/` directory, labeled by the epoch from which they were saved. Next, we perform Deep Feature Reweighting:
```
python dfr_step_2.py error-max --epoch 9
```
or to perform DFR on all checkpoints:
```
for i in {0..59}
do
    python dfr_step_2.py error-max --epoch $i
done
```

#### Section 4.3: Linearly separable perturbations are not necessary 

The following command tests the linear separability of [Adversarial Poison](https://arxiv.org/abs/2106.10807) perturbations:
```
python test_linear_separability.py error-max
```
and we can also test the linear separability of [Autoregressive Perturbations](https://proceedings.neurips.cc/paper_files/paper/2022/hash/af66ac99716a64476c07ae8b089d59f8-Abstract-Conference.html)

```
python test_linear_separability.py l2-ar
```

The linear separability (train accuracy) will be printed and a plot, named `test-linear-sep-{}.png`, is also saved to the repo directory showing the train accuracy of the logistic regression model over 500 iterations. 

#### Section 4.4: Orthogonal projection for learning from datasets with class-wise perturbations

The first step (Lines 1-4 of Algorithm 1) to using Orthogonal projection is training a logistic regression model. For example, let's train using [OPS Poison](https://github.com/cychomatica/One-Pixel-Shotcut).
```
python orthogonal_projection_step_1.py ops
```
The checkpoint is saved in `logistic-regression-ckpts/` directory along with a visualization of the recovered perturbations (weights from the linear model) in `ops.png`. The second step (Lines 5-6 of Algorithm 1) is to project the poisoned data and train a ResNet-18 model:
```
python orthogonal_projection_step_2.py ops
```
which results in a ResNet-18 with more than 87% test accuracy on CIFAR-10, despite using only OPS Poison training data.

### Citation

If you find this work useful for your research, please cite our paper:
```
@misc{sandovalsegura2023learn,
      title={What Can We Learn from Unlearnable Datasets?}, 
      author={Pedro Sandoval-Segura and Vasu Singla and Jonas Geiping and Micah Goldblum and Tom Goldstein},
      year={2023},
      eprint={2305.19254},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```