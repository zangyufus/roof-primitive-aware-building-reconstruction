# Roof-Primitive-Aware Building Reconstruction

## Overall Workflow

![Figure 1. Overall framework of the proposed roof-primitive-aware building reconstruction method.](assets/figure1.png)

*Figure 1. Overall framework of the proposed roof-primitive-aware building reconstruction method.*

## Introduction

![Method introduction](assets/introduction.png)

## Environment Requirements

The implementation runs on Linux or Ubuntu under WSL with Python 3.10, PyTorch 2.5.1, and CUDA runtime 12.4. Experiments were conducted on an NVIDIA RTX 3090 GPU with 24 GB of memory.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate roof-primitive
```

The [environment.yml](environment.yml) specifies package versions and installation sources.

## Data

**The full code and data will be uploaded after the paper is accepted.**

Dataset download: [Google Drive](https://drive.google.com/drive/folders/1HRBJz_U6rfd44xCufWqTk-m_oxhZZFt-?usp=sharing).

## Modeling Results

![Reconstructed building models in Tallinn.](assets/figure10_a.png)

*Tallinn.*

![Reconstructed building models on the NUIST campus.](assets/figure10_b.png)

*NUIST campus.*

![Reconstructed building models on the WHU campus.](assets/figure10_c.png)

*WHU campus.*


| Dataset | Mean vertices | Mean faces | Mean CD (cm) | RMSE (cm) | Edge precision (%) | Edge recall (%) | 2-Manifold (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tallinn | 66.40 | 47.50 | 23.98 | 10.42 | 91.12 | 89.79 | 97.26 |
| NUIST campus | 51.70 | 36.90 | 17.24 | 7.55 | 94.48 | 93.37 | 99.29 |
| WHU campus | 37.30 | 26.60 | 30.36 | 13.47 | 87.85 | 86.63 | 95.24 |
| Overall, macro-average across scenes | 51.80 | 37.00 | 23.86 | 10.48 | 91.15 | 89.93 | 97.26 |

Across the three scenes, the framework achieves a macro-averaged mean Chamfer distance of **23.86 cm** and RMSE of **10.48 cm**, with **37.00 faces per building**. Edge precision, edge recall, and 2-Manifold validity reach **91.15%**, **89.93%**, and **97.26%**, respectively, demonstrating compact reconstruction with accurate geometry and coherent connectivity.

Primitive decomposition achieves **91.63% AP** and **92.81% AR**, and six-category recognition achieves **93.04% overall accuracy**. Downloadable reconstructed models will accompany the full release after acceptance.

## Citation

Use the following entry to cite the manuscript. Publication metadata will be updated after acceptance.

```bibtex
@misc{zang_roof_primitive_reconstruction,
  title  = {Learn what and where; inherit how it is connected},
  author = {Zang, Yufu and Shi, Zhuokai and Chen, Dong and Guan, Haiyan and Yang, Bisheng},
  note   = {Manuscript; publication details will be updated after acceptance},
  url    = {https://github.com/zangyufus/roof-primitive-aware-building-reconstruction}
}
```
