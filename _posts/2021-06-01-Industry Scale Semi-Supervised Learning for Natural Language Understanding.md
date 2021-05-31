---
title: "Industry Scale Semi-Supervised Learning for Natural Language Understanding"
layout: post
date: 2021-06-01 01:00
image: /assets/images/markdown.jpg
headerImage: false
tag:
- Semi-Supervised Learning
- Natural Language Understanding
- Industry
- Virtual Adversarial Training

star: true
category: blog
author: dongju 
description: Industry Scale에서의 SSL 적용
usemath: true
toc: true
---

**Authors** : Luoxin Chen, Francisco Garcia, Varun Kumar, He Xie, Jianhua Lu - Alexa AI (amazon)
 
**NAACL2021**  
Paper :  [https://www.aclweb.org/anthology/2021.naacl-industry.39.pdf](https://www.aclweb.org/anthology/2021.naacl-industry.39.pdf){:target="_blank"}  
Code : -  

---


## Problem Definition

**SSL techniques in “real-world” NLU applications is still in question.**

1. Having high quality labeled data is the key to achieve improving accuracy.
    - However, obtaining human annotation is an expensive and time-consuming process.
2. A common practice to evaluate SSL algorithms is to take an existing labeled dataset and only use a small fraction of training data as labeled data, while treating the rest of the data as unlabeled dataset.
    - Such evaluation, often constrained to the cases when labeled data is scarce, raises questions about the usefulness of different SSL algorithms in a real-world setting.
3. How much unlabeled data should we use for SSL and how to select unlabeled data from a large pool of unlabeled data?  
4. Most SSL benchmarks make the assumption that unlabeled datasets come from the same distribution as the labeled datasets.
    - This assumption is often violated as, by design, the labeled training datasets also contain synthetic data, crowd-sourced data to represent anticipated usages of a functionality, and unlabeled data often contain a lot of out of domain data.

## Contribution

1. Design of a production SSL pipeline which can be used to intelligently select unlabeled data to train SSL models.
2. Experimental comparison of four SSL techniques including, Pseudo-Label, Knowledge Distillation, Cross-View Training, and Virtual Adversarial Training in a real-world setting using data from Amazon Alexa.
3. Operational recommendations for NLP practitioners who would like to employ SSL in production setting.

## Methods

<p align="center"><img src="{{site.url}}/{{site.post-assets}}/Industry Scale Semi-Supervised Learning for Natural Language Understanding/Untitled.png" width="100%" height="100%"> </p>


### Data Selection Approaches

1. First uses a classifier’s confidence score to filter domain specific unlabeled data from a very large pool of unlabeled data, which might contain data from multiple domains.
    - Training a binary classifier on the labelled data, and use it to select the in-domain unlabelled data.
    - Using Confidence scire 0.5 as the threshold for data selection.

2. The goal of the second stage filtering is to find a subset of data which could result in better performance in SSL training.

    **Selection by Submodular Optimization**

    (Paper : Submodularity for data selection in machine translation)

    - For SSL data selection, we use 1-4 n-gram as features and logarithm as the concave function.
    - We filter out any n-gram features which appear less than 30 times in Dl ∪ Du.
    - The algorithm starts with Dl as the selected data and chooses the utterance from the candidate pool Du which provides maximum marginal gain.

    **Selection by Committee**

    - To detect data points on which the model is not reliable, we train a committee of *n* teacher models (we use *n* = 4 in this paper), and compute the average entropy of the probability distribution for every data point.
    - Compute the average entropy of the predicted label distribution of *x.*
    - We then identify an entropy threshold with an acceptable error rate for mis-annotations (e.g., 20%) based on a held-out dataset.

### Semi-Supervised Learning Approaches

- Pseudo-labeling
- Knowledge Distillation
- Virtual Adversarial Training  
- [Cross-View Training (CVT) [LINK]](https://www.notion.so/Cross-View-Training-CVT-9501afc6c68c496bb2e2e7c8dda8e92d){:target="_blank"}

## Results

<p align="center"><img src="{{site.url}}/{{site.post-assets}}/Industry Scale Semi-Supervised Learning for Natural Language Understanding/Untitled%201.png" width="100%" height="100%"> </p>

<p align="center"><img src="{{site.url}}/{{site.post-assets}}/Industry Scale Semi-Supervised Learning for Natural Language Understanding/Untitled%202.png" width="50%" height="50%"> </p>

### Diversity of Selected Data

<p align="center"><img src="{{site.url}}/{{site.post-assets}}/Industry Scale Semi-Supervised Learning for Natural Language Understanding/Untitled%203.png" width="100%" height="100%"> </p>

- Measuring the diversity of the selected data by computing the unique n-gram ratio present in Dl ∪ Du and Dl data.
- We observe that a diverse SSL pool does not necessarily lead to better performance.
- This result highlights that simply optimizing for token diversity is not enough for improving SSL performance.

## Recommendations

1. Prefer VAT and CVT SSL techniques over PL and KL.
2. Use data selection to select a subset of unlabeled data.
    - Recommend Submodular Optimization based data selection in light of its lower cost and similar performance to committee based method.
    - Optimizing data selection, when unlabeled data pool is of a drastically different distribution from the labeled data, remains a challenge and could benefit from further research.

## Related Paper

[Self-training Improves Pre-training for Natural Language Understanding [LINK]](https://toriving.github.io/Self-training-Improves-Pre-training-for-Natural-Language-Understanding/){:target="_blank"}
