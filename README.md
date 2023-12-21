
# Quantization For All

We have implemented different quantization schemes including linear, non-linear, and fixed-grid-based binning for both weights and activations.

## Getting Started

You are required to have the basic knowledge of quantization to adopt any relevant technique and implement it for your use.
For hardware-specific tasks, fixed-grid-based quantization is better to adopt. For software-level quantization, one can choose any of the techniques provided above depending on the dynamic range of weight and activations.

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

