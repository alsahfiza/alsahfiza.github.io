---
layout: post
title: "Identify Metastatic Cancer"
date: 2019-12-02
excerpt_separator: <!--more-->
tags: [Data Science, Machine Learning, Deep Learning, Python, Healthcare]
image: "/images/perceptron/Metastatic-Cancer.png"
mathjax: "true"
---

In this project, we create an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans. The data for this competition is a slightly modified version of the PatchCamelyon (PCam) benchmark dataset. The original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on Kaggle does not contain duplicates.
<!--more-->


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
