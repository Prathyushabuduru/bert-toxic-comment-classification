# Toxic Comment Classification — BERT Fine-Tuning for Content Moderation

Fine-tuned a pre-trained BERT (`bert-base-uncased`) model using Hugging Face Transformers to detect toxic comments, benchmarked against a Logistic Regression baseline.

## Problem

Online platforms struggle to moderate toxic content because language nuances — sarcasm, implicit meaning, contextual dependencies — are hard for traditional ML models to capture. This project fine-tunes BERT for binary toxic/non-toxic classification and evaluates whether transformer-based contextual understanding meaningfully outperforms a classical baseline.

## Data

- **Source:** Jigsaw Toxic Comment Classification dataset (Kaggle)
- **Sample:** 15,000 comments (subset of the original 150K+)
- **Class balance:** ~90% non-toxic / 10% toxic
- **Split:** 80/20 train-validation

EDA showed comment length is highly right-skewed, and length alone doesn't separate toxic from non-toxic comments — reinforcing the need for a model that captures context rather than surface features.

## Modeling Approach

- **Model:** `bert-base-uncased`, fine-tuned via Hugging Face `Trainer` API
- **Tokenization:** padding + truncation, attention masks preserved for contextual relationships
- **Training:** 2 epochs, AdamW optimizer, learning rate 2e-5, with learning rate decay over training steps
- **Baseline:** Logistic Regression for comparison

## Results

| Model | Accuracy | F1 (toxic class) |
|---|---|---|
| Logistic Regression (baseline) | 93.9% | Lower — missed contextual cues |
| **BERT (fine-tuned)** | **96.5%** | **0.817** |

- Validation loss stabilized around **0.108**, indicating good convergence with minimal overfitting
- Confusion matrix showed low false positive/negative rates (53 FP, 53 FN out of ~3,000 validation samples)
- **Recall of 0.817** on the toxic class — roughly 82% of toxic comments correctly flagged

## Business Impact

- Higher recall reduces user exposure to harmful content
- High precision avoids over-flagging non-toxic comments, protecting user experience
- At platform scale (millions of comments/day), even a 5–10% detection improvement meaningfully cuts manual moderation workload and operational cost

## Limitations

- Class imbalance (90/10) may bias predictions toward the majority class
- Model classifies comments individually — no conversational context
- Potential bias in training data could affect fairness across demographic groups
- Future work: class weighting or resampling to improve minority-class detection

## Tech Stack

`Python` · `Hugging Face Transformers` · `BERT` · `AdamW Optimizer` · `Scikit-learn`

## Author

Prathyusha Buduru — M.S. Analytics, Saint Louis University
[LinkedIn](https://www.linkedin.com/in/prathyushabuduru/) · [GitHub](https://github.com/Prathyushabuduru)

## References

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*. https://arxiv.org/abs/1810.04805
- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*. https://arxiv.org/abs/1706.03762
- Kaggle. Jigsaw Toxic Comment Classification Challenge. https://www.kaggle.com/
