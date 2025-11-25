# Research Compass GNN

**AI-Powered Research Paper Topic Classification**

---

## Overview

Streamlit web application using Graph Neural Networks (GAT, GCN, GraphSAGE) to predict research paper topics from PDF uploads or text input.

### Key Features

- 🔮 **Multi-Model Support**: GAT, GCN, GraphSAGE architectures
- 📄 **PDF Processing**: Upload and analyze research papers automatically
- 🕸️ **Knowledge Graph**: Interactive citation network visualization
- 📊 **Topic Distribution**: Confidence scores for all predictions
- 🎯 **Dual Datasets**: OGB arXiv (169K CS papers) + AMiner (10K authors)

---

## Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/Apc0015/Rech_Comp.git
cd Rech_Comp

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

Access at: **http://localhost:8501**

### Streamlit Cloud Deployment

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your GitHub repo
4. Models and datasets auto-download on first run

---

## Datasets

| Dataset | Items | Classes | Description |
|---------|-------|---------|-------------|
| **OGB arXiv** | 169,343 | 40 | Computer Science papers from arXiv |
| **AMiner** | 10,000 | 10 | Author research field classification |

---

## How It Works

1. **Upload PDF** or paste paper text
2. **Select Dataset**: OGB arXiv (CS papers) or AMiner (authors)
3. **Choose Model**: GAT, GCN, or GraphSAGE
4. **Get Predictions**: Topic classification with confidence scores
5. **Explore Visualization**: Knowledge graph showing paper connections

---

## Models

### Graph Attention Network (GAT)
- 4 attention heads, 128 hidden dimensions
- Multi-head attention mechanism for paper features
- Best for capturing citation relationships

### Graph Convolutional Network (GCN)
- 128 hidden dimensions
- Spectral graph convolution
- Fast and efficient

### GraphSAGE
- 128 hidden dimensions  
- Neighbor sampling aggregation
- Scalable to large graphs

---

## Project Structure

```
Rech_Comp/
├── app.py                    # Main Streamlit application
├── saved_models/
│   ├── all_models.pt        # OGB arXiv trained models
│   └── aminer_models.pt     # AMiner trained models
├── notebooks/
│   ├── GNN_Complete_Standalone.ipynb    # OGB training
│   └── GNN_AMiner_Real_Fixed.ipynb      # AMiner training
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

**Note**: `data/` folder auto-downloads on first run and is excluded from git.

---

## Technical Stack

- **Streamlit** - Web application framework
- **PyTorch** - Deep learning framework
- **PyTorch Geometric** - GNN library
- **OGB** - Open Graph Benchmark datasets
- **Plotly** - Interactive visualizations
- **NetworkX** - Graph algorithms
- **PyPDF2** - PDF text extraction

---

## Training Your Own Models

See the notebooks in the `notebooks/` folder:

```bash
# For OGB arXiv dataset
jupyter notebook notebooks/GNN_Complete_Standalone.ipynb

# For AMiner dataset
jupyter notebook notebooks/GNN_AMiner_Real_Fixed.ipynb
```

Models are saved to `saved_models/` directory.

---

## License

MIT License

---

## References

- **GAT**: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (Veličković et al., 2018)
- **GCN**: [Semi-Supervised Classification with GCNs](https://arxiv.org/abs/1609.02907) (Kipf & Welling, 2017)
- **GraphSAGE**: [Inductive Representation Learning](https://arxiv.org/abs/1706.02216) (Hamilton et al., 2017)
- **OGB**: [Open Graph Benchmark](https://ogb.stanford.edu/)
