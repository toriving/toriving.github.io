---
title: "SSMBA: Self-Supervised Manifold Based Data Augmentation for Improving Out-of-Domain Robustness"
layout: post
date: 2020-12-09 02:19
image: /assets/images/markdown.jpg
headerImage: false
tag:
- Data augmentation
- Denoising Auto-Encoder
- Out-of-domain
star: true
category: blog
author: dongju 
description: Corruption Function과 Reconstruction Function을 정의해서 Denoising Auto-Encoder 방식으로 데이터를 생성
usemath: true
toc: true
---

 

**Authors** : Nathan Ng (University of Toronto Vector Institute), Kyunghyun Cho (New York University), Marzyeh Ghassemi (University of Toronto Vector Institute)

**EMNLP2020**
Paper : [https://www.aclweb.org/anthology/2020.emnlp-main.97.pdf](https://www.aclweb.org/anthology/2020.emnlp-main.97.pdf)
Code : [https://github.com/nng555/ssmba](https://github.com/nng555/ssmba)

---

## Summary

- Corruption Function과 Reconstruction Function을 정의해서 Denoising Auto-Encoder 방식으로 데이터를 생성
- Corruption function으로는 MLM을 Reconstruction function으로는 RoBERTa 사용
- Self-Supervision을 통해 생성된 데이터에 대해 레이블링
- Out-Of-Domain (OOD) 데이터에 대해서도 성능이 향상됨
- EDA / CBERT / UDA 와 비교

**개인적 견해**

- CBERT 이후에 나온 GPT2 / BART를 이용한 모델과 비교하지 않은 점이 아쉬움
- CBERT와 거의 비슷하지만 모델이 RoBRETa 로 바뀐 점, 그리고 레이블링을 하지 않는 점, 그리고 fine-tuning을 하지 않는 점이 다름
- CBERT에서는 가정이나 가설같은게 존재하지 않았는데, 이 논문에서는 그럴싸한 가설을 가져오고 가정을 함
- 다양한 분석이 Accept 요인이지 않을까?
- 어~~~디서 많이 본 것과 비슷한데....? (Ma....)

---

## Abstract

Models that perform well on a training domain often fail to generalize to out-of-domain (OOD) examples. Data augmentation is a common method used to prevent overfitting and improve OOD generalization. However, in natural language, it is difficult to generate new examples that stay on the underlying data manifold. We introduce SSMBA, a data augmentation method for generating synthetic training examples by using a pair of corruption and reconstruction functions to move randomly on a data manifold. We investigate the use of SSMBA in the natural language domain, leveraging the manifold assumption to reconstruct corrupted text with masked language models. In experiments on robustness benchmarks across 3 tasks and 9 datasets, SSMBA consistently outperforms existing data augmentation methods and baseline models on both in-domain and OOD data, achieving gains of 0.8% accuracy on OOD Amazon reviews, 1.8% accuracy on OOD MNLI, and 1.4 BLEU on in-domain IWSLT14 German-English.

examples by using a pair of corruption and re-

## 1. Introduction

- Training distributions (train set)은 test distribution을 전부 커버하지 못하는 경우가 많다.
- 이는 Biased dataset collection 또는 test distribution drift over time 으로 인해 나타난다.
- 따라서 unseen examples에 대해 강건하도록 학습시키는 것이 machine learning model 학습의 키포인트이다.
- 일반적으로 전체 분포로 일반화하는 것은 불가능 하므로 Out-Of-Domain (OOD) robustness 에 목표를 맞춘다.
- Data Augmentation (DA)는 OOD robustness를 향상시키는 일반적인 방법이다.
- 만약 데이터가 low-dimensional manifold에 집중되어 있다면, 그 데이터에 의해 DA된 데이터는 원래 데이터의 주변에 있어야 한다.
- 이러한 perturbation 방법론들 (DA) 는 semi-supervised and self-supervised settings에서 성능 향상 또는 일반화가 되는걸 보여주었다.
- 이미지 데이터는 회전이나 간단한 transformation을 통해 DA가 가능하지만, NLP는 의미를 보존하면서 DA 하기가 힘들다.
- 본 논문에서는 Self-Supervised Manifold Based Data Augmentation (SSMBA)를 제안한다.
- SSMBA는 휴리스틱하게 특성화하기 어려운 도메인에서 DA를 하는 방법이다.
    - 휴리스틱한 방법론으로 NLP는 DA하기 어렵다는 뜻
- Denoising auto-encoder을 모티브로함
    - Corruption function을 통해 data manifold에서 확률적으로 examples을 *off* (perurb) 한다.
    - 그 다음 Reconstruction function을 통해 *back on* (project) 한다

        ![SSMBA%20Self-Supervised%20Manifold%20Based%20Data%20Augmenta%205c514340a5504ac286e24639e1e90a4b/Untitled.png](SSMBA%20Self-Supervised%20Manifold%20Based%20Data%20Augmenta%205c514340a5504ac286e24639e1e90a4b/Untitled.png)

    - 이런 방식으로 하면 DA 된 데이터가 원래 데이터의 주변에 놓이게 된다.
- SSMBA는 모든 supervised task에 적용할 수 있으며, task-specific한 knowledge가 필요하지 않고, class- 또는 dataset-specific fine-tuning이 필요하지 않다.
- 감정 분석, 자연어 추론 및 기계 번역에서 SSMBA를 사용하는 방법을 조사함
- 9개의 데이터 세트와 4개의 모델에 대한 실험에서 SSMBA가 In-domain 및 OOD 데이터 모두에서 baseline과 다른 방법론보다 좋음

## 2. Background and Related Work

### 2. 1. Data Augmentation in NLP

- 과거 연구들은 일반화를 개선하기 위한 방법으로 DA을 사용하였으며, 기존 데이터를 바탕으로 새로운 데이터를 만드는 방식을 사용했다
- Semi-supervised learning 및 self-supervised learning settings에서 효과적인 이러한 방법들은 local perturbation에 대한 robustness를 유도한다고 가정한다.
- Task-specific한 방법들이 존재하는데, 예를 들면 휴리스틱 기반, back-translation, consistency training, word embedding을 이용하는 방법, Language model을 이용하는 방법이 있다.
- 최근에는 PLM을 이용하여 샘플을 생성하거나, contextual language model을 finetuning하여 사용하는 방법이 있다.

### 2. 2. VRM and the Manifold Assumption

- Vicinal Risk Minimization (VRM)은 기존 training data 주변에서 샘플을 추출하여 training data를 확대하는 것으로 DA를 정의한다.
    - VRM에 대해 알아보자
    - 일반적으로 우리에게 주어진 데이터를 가지고 학습하는 방식을 Empirical Risk Minimization (ERM) 이라고 한다.
    - 이 Empirical data의 주변부(vicinity) 분포를 모델링하여 이 Vicinal data distribution에 대해 학습하는 것을 VRM이라고 한다.
    - 즉 기존 데이터에 local perturbation을 주는 방식을 Vicinity 분포를 모델링하는 방식이라고 볼 수 있다.
    - 따라서 DA via local perturbation 를 사용한 학습은 VRM이라고 할 수 있다.
- 일반적으로 training data의 주변은 데이터 세트에 따른 휴리스틱을 사용하여 정의함.
- Computer Vision 분야에서는 scaling을 하거나 color를 변경하거나 translation, roation을 하는 방식.

- Manifold assumption은 고차원 데이터가 저차원 manifold에 놓여 있다는 것을 말함.
- 이 가정을 통해 trainining example의 주변부를 data manifold에 있는 주변부분인 manifold neighborhood로 정의할 수 있다
    - This assumption allows us to define the vicinity of a training example as its manifold neighborhood, the portion of the neighborhood that lies on the data manifold.
- 최근 Manifold assumption을 바탕으로 decision boundary를 확장하거나, adversarial example을 생성하거나, 두개의 example을 interpolation하는 방식 또는 affine transformation을 하여 robustness를 증가시켰다.

### 2. 3. Sampling from Denoising Autoencoders

### 2. 4. Masked Language Models

## 3. SSMBA: Self-Supervised Manifold Based Augmentation

## 4. Datasets

### 4.1. Sentiment Analysis

### 4.2. Natural Language Inference

### 4.3. Machine Translation

## 5. Experimental Setup

### 5. 1. Model Types

### 5. 2. SSMBA Settings

### 5. 3. Baselines

### 5. 4. Evaluation Method

## 6. Results

### 6. 1. Sentiment Analysis

### 6. 2. Natural Language Inference

### 6. 3. Machine Translation

## 7. Analysis and Disccusion

### 7. 1. Training Set Size

### 7. 2. Reconstruction Model Capacity

### 7. 3. Corruption Amount

### 7. 4. Sample Generation Methods

### 7. 5. Amount of Augmentation

### 7. 6. Label Generation

## 8. Conclusion

> In this paper, we introduce SSMBA, a method for generating synthetic data in settings where the underlying data manifold is difficult to char- acterize. In contrast to other data augmentation methods, SSMBA is applicable to any supervised task, requires no task-specific knowledge, and does not rely on dataset-specific fine-tuning. We demonstrate SSMBA’s effectiveness on three NLP tasks spanning classification and sequence mod- eling: sentiment analysis, natural language infer- ence, and machine translation. We achieve gains of 0.8% accuracy on OOD Amazon reviews, 1.8% accuracy on OOD MNLI, and 1.4 BLEU on in- domain IWSLT14 de!en. Our analysis shows that SSMBA is robust to the initial dataset size, recon- struction model choice, and corruption amount, offering OOD robustness improvements in most set- tings. Future work will explore applying SSMBA to the target side manifold in structured prediction tasks, as well as other natural language tasks and settings where data augmentation is difficult.

---
