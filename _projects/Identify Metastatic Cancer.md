---
# file: _projects/Identify Metastatic Cancer.md
layout:      project
title:       Identify Metastatic Cancer*
date:        12 Feb 2019
image:
  path:       /assets/projects/Metastatic-Cancer.png
  srcset:
    1920w:   /assets/projects/Metastatic-Cancer.png
    960w:    /assets/projects/Metastatic-Cancer.png
    480w:    /assets/projects/Metastatic-Cancer.png
# caption:     Hyde is a brazen two-column Jekyll theme.
description: >
              In this project, we create an algorithm to identify metastatic cancer in small image patches taken from larger digital 
              pathology scans. The data for this competition is a slightly modified version of the PatchCamelyon (PCam) benchmark 
              dataset.
#links:
#  - title:   Link
#    url:     _posts/2019-12-02-MetastaticCancer.md
featured:    false
---
The original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on Kaggle does not contain duplicates.


Dataset Links [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/data)


## Model Plot
![png](/images/MetastaticCancer/model_plot.png)

## Training and validation Loss
![png](/images/MetastaticCancer/training1.png)

## Training and validation Accuracy
![png](/images/MetastaticCancer/validation1.png)

## Receiver operating characteristic (roc) curve
![png](/images/MetastaticCancer/roc1.png)

## Classification Report
![png](/images/MetastaticCancer/ClassificationReport.png)

## Confusion Matrix
![png](/images/MetastaticCancer/cmatrix1.png)
