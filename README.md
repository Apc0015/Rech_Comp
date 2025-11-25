# Research Compass GNN

**Graph Neural Networks for Citation Network Analysis**

---

## Overview

Implementation of Graph Neural Networks (GAT, GCN, GraphSAGE) for citation network analysis:

- Node Classification: Paper topic classification
- Link Prediction: Citation prediction
- Graph Visualization: Network structure exploration
- Attention Visualization: GAT attention mechanism analysis

---

## Features

- Multiple GNN architectures: GAT, GCN, GraphSAGE
- Web-based interface (Gradio)
- Standard benchmark datasets: Cora, CiteSeer, PubMed, OGB arXiv
- Interactive visualizations and attention analysis
- Real-time training progress monitoring

---

## Installation

### Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install PyTorch Geometric dependencies (CPU version)
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Running the Application

```bash
python app.py
```

### Access

Go to: **http://localhost:7861**

---

## Datasets

| Dataset | Papers | Classes | Topic |
|---------|--------|---------|-------|
| **Cora** | 2,708 | 7 | Machine Learning |
| **CiteSeer** | 3,327 | 6 | Computer Science |
| **PubMed** | 19,717 | 3 | Medical Research |
| **OGB arXiv** | 169,343 | 40 | Computer Science (arXiv papers) |

---

## Interface

### Load Dataset
- Built-in datasets or custom upload (CSV, JSON, BibTeX, PDF)
- Dataset selection and configuration

### Train Models
- Task selection (Node Classification or Link Prediction)
- Model selection (GAT, GCN, GraphSAGE)
- Hyperparameter configuration
- Real-time training metrics
- Performance comparison

### Visualizations
- Training curves
- Confusion matrices
- Interactive network graphs
- ROC curves (Link Prediction)
- Precision-Recall curves (Link Prediction)

---

## Models

### Graph Attention Network (GAT)
- Multi-head attention mechanism
- Attention weight visualization
- Cora accuracy: ~83%

### Graph Convolutional Network (GCN)
- Spectral graph convolution
- Efficient training
- Cora accuracy: ~81%

### GraphSAGE
- Neighbor sampling aggregation
- Inductive learning capability
- Cora accuracy: ~80%

---

## Project Structure

```
Research-Compass-GNN/
├── app.py                     # Main Gradio UI application
├── models/
│   ├── gat.py                 # Graph Attention Network model
│   ├── gcn.py                 # Graph Convolutional Network model
│   └── graphsage.py           # GraphSAGE model
├── data/
│   ├── dataset_utils.py       # Load citation datasets
│   └── multi_format_processor.py  # Handle multiple data formats
├── training/
│   └── trainer.py             # Training infrastructure (BaseTrainer, GCNTrainer)
├── evaluation/
│   ├── metrics.py             # Accuracy, F1, etc.
│   └── visualizations.py      # Confusion matrix, plots
├── visualization/
│   ├── graph_viz.py           # Network visualization
│   └── attention_viz.py       # Attention heatmaps
├── utils/
│   └── checkpoint.py          # Model checkpointing utilities
├── config/
│   └── settings.py            # Configuration manager
├── tools/
│   └── verification/          # Verification scripts
├── docs/
│   └── archive/               # Historical documentation
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Architecture

### Graph Representation
- Nodes: Research papers
- Edges: Citation relationships
- Features: Text embeddings (TF-IDF or Sentence-BERT)
- Labels: Paper topic categories

### Training Pipeline

**Node Classification:**
1. Dataset loading and preprocessing
2. Model initialization
3. Forward propagation through GNN layers
4. Loss computation (Cross-Entropy)
5. Backpropagation and optimization (Adam)
6. Validation and testing

**Link Prediction:**
1. Dataset loading and edge split creation
2. Model initialization
3. Forward propagation and edge prediction
4. Loss computation (Binary Cross-Entropy)
5. Backpropagation and optimization (Adam)
6. Validation and testing (ROC-AUC, Average Precision)

---

## Performance

| Dataset | Test Accuracy | Classes | Training Time |
|---------|--------------|---------|---------------|
| Cora | ~83% | 7 | 1-2 min |
| CiteSeer | ~72% | 6 | 1-2 min |
| PubMed | ~79% | 3 | 3-5 min |

---

## Technical Stack

### Technology Stack

- **PyTorch** - Deep learning framework
- **PyTorch Geometric** - Graph neural network library
- **Gradio** - Interactive web UI
- **NetworkX** - Graph algorithms
- **Matplotlib/Plotly** - Visualization

### Model Architectures

```python
# Graph Attention Network (GAT)
GAT(
  input_dim=1433,        # Paper features
  hidden_dim=128,        # Hidden layer size
  output_dim=7,          # Number of classes
  heads=8,               # Attention heads (GAT-specific)
  dropout=0.3            # Regularization
)

# Graph Convolutional Network (GCN)
GCN(
  input_dim=1433,        # Paper features
  hidden_dim=128,        # Hidden layer size
  output_dim=7,          # Number of classes
  dropout=0.3            # Regularization
)

# GraphSAGE
GraphSAGE(
  input_dim=1433,        # Paper features
  hidden_dim=128,        # Hidden layer size
  output_dim=7,          # Number of classes
  aggregator='mean'      # Neighbor aggregation method
)
```

### Training Configuration

- **Optimizer**: Adam
- **Learning Rate**: 0.01
- **Epochs**: 100
- **Loss**: Cross-Entropy (Node Classification), BCEWithLogits (Link Prediction)
- **Metrics**: Accuracy, Precision, Recall, F1 (Node Classification); ROC-AUC, Average Precision (Link Prediction)

---

## Implementation Details

- Graph neural network architectures
- Attention mechanisms (GAT)
- Node classification and link prediction
- Model training and evaluation
- Network visualization techniques

---

## Troubleshooting

**PyTorch Geometric dependencies:**
```bash
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**Port conflicts:** Default port 7861. Check terminal output for alternative port.

**Public Share Links:** The app creates a public Gradio share link (`share=True`) to ensure accessibility even when localhost has connectivity issues. This link expires in 72 hours.

---

## References

1. **GAT Paper**: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (Veličković et al., 2018)
2. **GCN Paper**: [Semi-Supervised Classification with GCNs](https://arxiv.org/abs/1609.02907) (Kipf & Welling, 2017)
3. **GraphSAGE Paper**: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (Hamilton et al., 2017)

### Datasets

- **Cora/CiteSeer/PubMed**: Standard citation network benchmarks
- From [PyTorch Geometric Datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)

---

## License

MIT License
