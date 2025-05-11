# Short-term prediction of solar radiation from sky photographs using new trends in machine learning methods

## Diploma Thesis

[![Python 3.10](https://img.shields.io/badge/Python-3.10-brightgreen?style=plastic)](https://www.python.org/downloads/release/python-3102/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=plastic)](https://github.com/PeterKoXcode/solarVision/blob/main/LICENSE)
[![torch](https://img.shields.io/badge/torch-2.6.0%2Bgpu-orange?style=plastic)](https://pytorch.org/docs/stable/index.html)
[![SkyCam](https://img.shields.io/badge/Dataset-SkyCam-blueviolet?style=plastic)](https://github.com/vglsd/SkyCam)
[![Author](https://img.shields.io/badge/Author-Bc._Peter_Kopecký-blue?style=plastic)](https://is.stuba.sk/lide/clovek.pl?id=111310;)

This repository contains the source files of my diploma thesis developed in 2025 at the Faculty of Electrical
Engineering and Information Technology, Slovak University of Technology in Bratislava.

## About

This diploma thesis addresses short-term solar irradiance prediction from sky images
using the latest trends in machine learning methods, specifically Transformers architecture. Goal is to increase the
efficiency of solar farm management, thereby minimizing fluctuations in
production — whether excesses or shortages of electric energy.

## Dataset

Developed models work with a combination of image data and numerical meteorological data
obtained from the Swiss research center. The dataset called [SkyCam](https://github.com/vglsd/SkyCam) and
individual data samples are collected exclusively at 15-minute intervals, which is ideal for short-term
prediction, as electricity is typically traded on the market in the same time step.

## Code explanation

- **`vit_torch.py`**  
  Implements a standalone ViT model that predicts irradiance from sequences of sky images only.

- **`vit_torch_num.py`**  
  Implements a combined model that uses both sky image sequences and associated numerical meteorological data (
  Irradiance, Zenith, Temperature, Humidity, Pressure, Hour) for improved prediction accuracy.

## Install dependencies:

```bash
    pip install torch torchvision numpy pandas opencv-python matplotlib scikit-learn
```

## Getting started

Follow these steps to set up and run the project:

1. Clone the repository using password-protected SSH key:

```bash
    git clone git@github.com:PeterKoXcode/solarVision.git
    cd solarVision/
```

2. Create and activate a virtual environment (optional but recommended):

```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Download the dataset SkyCam and save the folder indicating the name of the location in the `../datasets/tsi_dataset/`
   directory. For example: `../datasets/tsi_dataset/expo10_Alpnach2018/`.
4. Run the Python files in **models** folder.

## Conclusion

This research has demonstrated the potential of an unconventional approach, but it
could not cover all possible experiments — such as implementing LSTM, training on a
different dataset, or applying our data to another existing model. Therefore, we
believe this work has strong potential for improvement and can serve as a valuable
foundation for future extensions and research.
