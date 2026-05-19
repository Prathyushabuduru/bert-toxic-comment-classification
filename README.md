# Toxic Comment Classification — Fine-Tuned BERT

## Overview
Fine-tuned BERT (bert-base-uncased) for binary toxic comment 
classification using the Jigsaw dataset. Achieved 96.5% accuracy 
and F1-score of 0.817, outperforming Logistic Regression baseline.

## Problem Statement
Online platforms need automated systems to detect harmful content 
at scale. Traditional ML models fail to capture contextual language 
nuances — this project uses transformer-based BERT to solve that.

## Dataset
- Source: Jigsaw Toxic Comment Classification (Kaggle)
- Size: 15,000 comments sampled from 150,000+
- Class distribution: 90% non-toxic, 10% toxic

## Tech Stack
- Python
- HuggingFace Transformers
- BERT (bert-base-uncased)
- PyTorch
- Pandas, NumPy, Matplotlib

## Approach
1. Exploratory Data Analysis
2. Tokenization with padding and truncation
3. Fine-tuning BERT with AdamW optimizer (lr=2e-5)
4. 80/20 train/validation split
5. Evaluation using Accuracy, F1-score, Confusion Matrix

## Results
| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression | 93.9% | Lower |
| Fine-Tuned BERT | 96.5% | 0.817 |

## Business Impact
- 82% of toxic comments correctly identified
- Reduces manual moderation workload at scale
- Applicable to platforms handling millions of daily comments

## How to Run
1. Clone this repository
2. Install requirements: pip install -r requirements.txt
3. Open the notebook: jupyter notebook Week_6_Group_1.ipynb
