
# Quantization For All
We have implemented different quantization schemes including linear, non-linear, and fixed-grid-based binning for both weights and activations.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contact](#contact)
- [Papers](#papers)

---

## Features
- **Ultra-Low Precision Quantization**: Employs gird-based quantization techniques tailored for hardware-aware spiking neural networks.
- **Scalability**: Demonstrates high performance across various CNN/SNN architectures and event/static datasets.

---

## Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.9
- CUDA (Optional, for GPU acceleration)

### Install Required Packages
Clone the repository and install dependencies:

```bash
git clone https://github.com/Ahmedhasssan/Quantization_for_all.git
cd Quantization_for_all
pip install -r requirements.txt
```

## Usage
### Getting Started

You are required to have the basic knowledge of quantization to adopt any relevant technique and implement it for your use.
For hardware-specific tasks, fixed-grid-based quantization is better to adopt. For software-level quantization, one can choose any of the techniques provided above depending on the dynamic range of weight and activations.

### Example Scripts
The repository includes examples for training and evaluating SNNs on popular Event (**DVS-MNIST**, **DVS-CIFAR-10**) and Static image (**MNIST**, **CIFAR-10** and **Caltech-101**) datasets:

```bash
bash scripts/vgg9_dvs_cifar.sh
bash scripts/vgg9_dvs_caltech.sh
```

## Results

## Contact

For any inquiries or collaboration opportunities, feel free to reach out:

- **Email**: [ah2288.@cornell.edu](mailto:ah2288@cornell.edu)
- **GitHub**: [Ahmedhasssan](https://github.com/Ahmedhasssan)

## Papers

Papers that used this repository:

1. **IM-SNN**: Hasssan, A., Meng, J., Anupreetham, A., & Seo, J. S. (2024, August). IM-SNN: Memory-Efficient Spiking Neural Network with Low-Precision Membrane Potentials and Weights. IEEE/ACM International Conference on Neuromorphic Systems (ICONS).*. [Link to paper](https://par.nsf.gov/biblio/10545833).
2. **Sp-QuantSNN**: Hasssan, Ahmed, Jian Meng, Anupreetham Anupreetham, and Jae-sun Seo. "SpQuant-SNN: ultra-low precision membrane potential with sparse activations unlock the potential of on-device spiking neural networks applications." Frontiers in Neuroscience 18 (2024): 1440000. [Link to paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/full).
3. 


We welcome feedback, suggestions, and contributions to enhance SpQuant-SNN!





### Prerequisites

1. GPU is required for training.
2. Pytorch >1.9

### Installing

1. Install the requirements by running "pip install -r requirements.txt"



## Built With

* [PACT Qunat](https://arxiv.org/abs/1805.06085) 
* [SAWB Quant](https://arxiv.org/abs/1807.06964)
* [Log-2 Quant](https://arxiv.org/abs/2203.05025)
* [AdRound Quant](https://arxiv.org/pdf/2004.10568.pdf)
* [Straigh Through Aproxx](https://openreview.net/pdf?id=Skh4jRcKQ)
 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

