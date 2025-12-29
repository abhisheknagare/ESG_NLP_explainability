# Benchmarking Post-Hoc Explainability for ESG Text Classification

A systematic comparison of SHAP, LIME, and attention mechanisms for explaining DistilBERT predictions on Environmental, Social, and Governance (ESG) text classification.

## Overview

This project benchmarks three post-hoc explainability methods for ESG text analysis:
- **SHAP** (SHapley Additive exPlanations) - Game-theoretic causal attributions
- **LIME** (Local Interpretable Model-agnostic Explanations) - Local linear approximations
- **Attention Mechanisms** - Transformer attention weights

We evaluate these methods on a binary classification task distinguishing substantive (quantitative) from vague (qualitative) ESG disclosures using a fine-tuned DistilBERT model.

## Key Findings

- **SHAP produces the most faithful explanations** with probability drops of Δp = 0.12 when top tokens are removed (vs. 0.09 for LIME, 0.07 for attention)
- **Methods show moderate agreement** (Spearman ρ ≤ 0.42), indicating complementary perspectives
- **SHAP emphasizes quantitative cues** (numbers, percentages, years), LIME highlights semantic phrases, attention focuses on content words
- **Runtime tradeoff**: Attention (0.01s) enables real-time use, LIME (2.4s) suits exploratory analysis, SHAP (6.7s) for high-stakes decisions

## Dataset

**Source**: ./esg_data.xlsx
Dataset included in this repository: esg_data.xlsx

- 7,988 ESG-relevant sentences from Chinese corporate sustainability reports (English-translated)
- 36 ESG topic categories organized under Environmental, Social, and Governance domains
- Binary quality labels: Substantive (26.3%) vs. Vague (73.7%)
- Stratified 70/15/15 train-dev-test split

## Requirements

```
python>=3.8
torch>=1.10.0
transformers>=4.20.0
shap>=0.41.0
lime>=0.2.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Usage

### 1. Data Preprocessing

```python
python preprocess.py --input data/raw/esg_dataset.csv --output data/processed/
```

Filters irrelevant sentences, aggregates 36 topics into 3 ESG domains, remaps quality labels, and creates stratified splits.

### 2. Train DistilBERT Classifier

```python
python train.py --config configs/distilbert_config.yaml
```

Fine-tunes `distilbert-base-uncased` with class-weighted cross-entropy, AdamW optimizer, and early stopping.

**Model Performance**:
- Test Accuracy: 0.92
- Macro F1: 0.91
- Substantive F1: 0.87

### 3. Generate Explanations

```python
python explain.py --model checkpoints/best_model.pt --method shap --samples 50
```

Options for `--method`: `shap`, `lime`, `attention`, or `all`

### 4. Evaluate Faithfulness

```python
python evaluate_faithfulness.py --explanations results/explanations/ --top_k 1,3,5
```

Performs deletion tests by masking top-k tokens and measuring probability drops.

### 5. Analyze Agreement

```python
python evaluate_agreement.py --explanations results/explanations/
```

Computes pairwise Spearman rank correlations between methods.

### 6. Visualize Results

```python
python visualize.py --results results/ --output figures/
```

Generates comparison plots, confusion matrices, and case study visualizations.

## Project Structure

```
esg-explainability-benchmark/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Preprocessed splits
├── configs/
│   └── distilbert_config.yaml  # Model hyperparameters
├── src/
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── models.py               # DistilBERT classifier
│   ├── explainers.py           # SHAP, LIME, Attention implementations
│   ├── evaluation.py           # Faithfulness and agreement metrics
│   └── visualization.py        # Plotting utilities
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── case_studies.ipynb
├── results/                    # Generated explanations and metrics
├── figures/                    # Visualization outputs
├── checkpoints/                # Trained model weights
├── preprocess.py
├── train.py
├── explain.py
├── evaluate_faithfulness.py
├── evaluate_agreement.py
├── visualize.py
├── requirements.txt
└── README.md
```

## Evaluation Metrics

### Faithfulness (Causal Alignment)
Measures whether highly-ranked tokens genuinely drive predictions through deletion tests:
- Remove top-k tokens (k ∈ {1, 3, 5})
- Replace with [MASK]
- Compute probability drop: Δp = p_original - p_masked

### Agreement (Inter-method Consistency)
Spearman rank correlation (ρ) between token importance rankings across methods.

### Runtime Analysis
Average per-instance processing time on CPU hardware (Intel i7-10700K, 32GB RAM).

## Practical Recommendations

**High-stakes decisions** (regulatory audits, due diligence): Use SHAP for maximum faithfulness

**Exploratory analysis** (initial screening, hypothesis generation): Use LIME for balanced speed and interpretability

**Real-time screening** (processing thousands of reports): Use attention for instant feedback

**Divergent explanations**: When methods disagree significantly, flag for human expert review

## Limitations

- Single dataset (Chinese-to-English translations)
- Binary classification only
- One model architecture (DistilBERT 66M parameters)
- Automated faithfulness evaluation (no human expert validation)
- Class imbalance (73.7% vague vs. 26.3% substantive)

## Future Work

- Extend to multilingual corpora and multi-label hierarchical classification
- Human-centered evaluation with ESG analysts
- Hybrid explainability methods (attention-guided SHAP, ensemble approaches)
- Adaptation for large language models (GPT-4, Claude, Llama)
- Longitudinal analysis tracking ESG language evolution
