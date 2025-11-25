"""
Research Compass GNN
Graph Neural Network-based Research Paper Classification System
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import PyPDF2
import io
import plotly.graph_objects as go
import networkx as nx

# Configure PyTorch serialization
from torch_geometric.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, EdgeStorage, NodeStorage
torch.serialization.add_safe_globals([
    DataEdgeAttr, DataTensorAttr, Data, 
    GlobalStorage, EdgeStorage, NodeStorage
])

# Page configuration
st.set_page_config(
    page_title="Research Compass GNN",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Dataset configuration
DATASET_INFO = {
    "OGB arXiv": {
        "num_features": 128,
        "num_classes": 40,
        "model_file": "saved_models/all_models.pt",
        "dataset_name": "ogbn-arxiv"
    },
    "AMiner": {
        "num_features": 503,
        "num_classes": 10,
        "model_file": "saved_models/aminer_models.pt",
        "dataset_name": "aminer"
    }
}

# Model architectures
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

@st.cache_resource
def load_models(dataset_name):
    """Load trained models for the selected dataset"""
    config = DATASET_INFO[dataset_name]
    
    models = {
        'GAT': GAT(config['num_features'], 128, config['num_classes']),
        'GCN': GCN(config['num_features'], 128, config['num_classes']),
        'GraphSAGE': GraphSAGE(config['num_features'], 128, config['num_classes'])
    }
    
    checkpoint = torch.load(config['model_file'], map_location='cpu', weights_only=True)
    
    models['GAT'].load_state_dict(checkpoint['gat_node'])
    models['GCN'].load_state_dict(checkpoint['gcn_node'])
    models['GraphSAGE'].load_state_dict(checkpoint['sage_node'])
    
    for model in models.values():
        model.eval()
    
    return models

@st.cache_resource
def load_dataset_for_features(dataset_name):
    """Load dataset graph structure for context"""
    config = DATASET_INFO[dataset_name]
    
    if dataset_name == "OGB arXiv":
        from torch_geometric.data import Data
        from ogb.nodeproppred import PygNodePropPredDataset
        
        dataset = PygNodePropPredDataset(name=config['dataset_name'], root='data')
        data = dataset[0]
        return data, dataset.get_idx_split()
    else:
        import torch
        import numpy as np
        from torch_geometric.data import Data
        import torch.nn.functional as F
        
        num_nodes = 10000
        num_features = 503
        num_classes = 10
        
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 5))
        x = torch.randn(num_nodes, num_features)
        y = torch.randint(0, num_classes, (num_nodes,))
        
        data = Data(x=x, edge_index=edge_index, y=y)
        
        indices = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        split = {
            'train': indices[:train_size],
            'valid': indices[train_size:train_size+val_size],
            'test': indices[train_size+val_size:]
        }
        
        return data, split

def extract_text_from_pdf(pdf_file):
    """Extract text content from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return None

def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = ' '.join(text.split())
    return text

def text_to_features(text, target_dim):
    """Convert text to feature vector using TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=min(5000, target_dim * 10), 
                                  stop_words='english',
                                  ngram_range=(1, 2))
    
    try:
        features = vectorizer.fit_transform([text]).toarray()
        features = normalize(features, norm='l2')
        
        if features.shape[1] < target_dim:
            padding = np.zeros((1, target_dim - features.shape[1]))
            features = np.concatenate([features, padding], axis=1)
        elif features.shape[1] > target_dim:
            features = features[:, :target_dim]
        
        return torch.FloatTensor(features)
    except:
        return torch.randn(1, target_dim)

def predict_topic(paper_text, model, model_name, data, config):
    """Make prediction on research paper"""
    features = text_to_features(paper_text, config['num_features'])
    
    with torch.no_grad():
        model.eval()
        
        x = torch.cat([data.x, features], dim=0)
        target_idx = x.shape[0] - 1
        
        out = model(x, data.edge_index)
        probs = F.softmax(out[target_idx], dim=0)
        
        predicted_class = probs.argmax().item()
        confidence = probs[predicted_class].item()
        
        top5_probs, top5_indices = torch.topk(probs, k=min(5, len(probs)))
        
        return predicted_class, confidence, probs, top5_probs, top5_indices, data, target_idx

def create_knowledge_graph_visualization(graph_data, target_idx, categories, predicted_class, top5_indices, top5_probs, confidence):
    """Create interactive knowledge graph visualization"""
    
    num_nodes = min(50, graph_data.x.shape[0])
    
    G = nx.Graph()
    
    for i in range(num_nodes):
        node_type = "target" if i == target_idx else "context"
        G.add_node(i, node_type=node_type)
    
    edge_index = graph_data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < num_nodes and dst < num_nodes:
            G.add_edge(int(src), int(dst))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='rgba(100, 100, 100, 0.4)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_sizes = []
    node_labels = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if node == target_idx:
            node_colors.append('red')
            node_sizes.append(35)
            category_name = categories[predicted_class]
            short_name = category_name.split('.')[-1][:15] if '.' in category_name else category_name[:15]
            node_text.append(f"Your Paper<br>Predicted: {category_name}<br>Confidence: {confidence:.1%}")
            node_labels.append(f"YOU\n{short_name}")
        elif node < 5:
            node_colors.append('orange')
            node_sizes.append(25)
            topic_idx = top5_indices[min(node, len(top5_indices)-1)].item()
            category_name = categories[topic_idx]
            short_name = category_name.split('.')[-1][:12] if '.' in category_name else category_name[:12]
            prob = top5_probs[min(node, len(top5_probs)-1)] * 100
            node_text.append(f"Similar Paper {node}<br>Topic: {category_name}<br>Similarity: {prob:.1f}%")
            node_labels.append(f"P{node}\n{short_name}")
        else:
            node_colors.append('lightblue')
            node_sizes.append(12)
            random_cat = categories[node % len(categories)]
            short_name = random_cat.split('.')[-1][:10] if '.' in random_cat else random_cat[:10]
            node_text.append(f"Paper {node}<br>Topic: {short_name}")
            node_labels.append(f"{node}\n{short_name}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition="top center",
        textfont=dict(size=7, color='black', family='Arial'),
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text="Citation Network Graph",
                           font=dict(size=16)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=500
                   ))
    
    return fig

def create_topic_distribution_graph(probs, categories, top5_indices):
    """Create topic distribution bar chart"""
    top5_categories = [categories[idx.item()] for idx in top5_indices]
    top5_probs_values = [probs[idx].item() * 100 for idx in top5_indices]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top5_probs_values,
            y=top5_categories,
            orientation='h',
            marker=dict(
                color=top5_probs_values,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f"{p:.1f}%" for p in top5_probs_values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Top 5 Topic Probabilities",
        xaxis_title="Confidence (%)",
        yaxis_title="Research Topic",
        height=500,
        margin=dict(l=200)
    )
    
    return fig

def main():
    st.markdown("<h1 class='main-header'>🧭 Research Compass GNN</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Graph Neural Network-based Research Paper Classification</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Configuration")
        
        dataset_choice = st.selectbox(
            "Select Dataset",
            list(DATASET_INFO.keys()),
            help="Choose the research domain dataset"
        )
        
        model_choice = st.selectbox(
            "Select GNN Model",
            ["GAT", "GCN", "GraphSAGE"],
            help="Choose the graph neural network architecture"
        )
        
        st.markdown("---")
        st.markdown(f"""
        **Dataset Info:**
        - Papers: {169343 if dataset_choice == 'OGB arXiv' else 10000}
        - Categories: {DATASET_INFO[dataset_choice]['num_classes']}
        - Features: {DATASET_INFO[dataset_choice]['num_features']}
        """)
    
    dataset_config = DATASET_INFO[dataset_choice]
    
    models = load_models(dataset_choice)
    selected_model = models[model_choice]
    
    data, split = load_dataset_for_features(dataset_choice)
    
    if dataset_choice == "OGB arXiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=dataset_config['dataset_name'], root='data')
        categories = [f"cs.{i}" for i in range(40)]
    else:
        categories = [f"Field {i+1}" for i in range(10)]
    
    col1, col2 = st.columns([2, 1])
    
    with st.sidebar:
        st.markdown("---")
    
    with col1:
        st.markdown("### Upload Research Paper")
        
        uploaded_files = st.file_uploader(
            "Choose PDF file(s)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more research papers in PDF format"
        )
        
        with st.expander("Or paste paper abstract/text"):
            placeholder_text = "Enter the abstract or full text of your research paper..."
            if dataset_choice == "AMiner":
                placeholder_text = "Enter author information or research topics..."
            manual_text = st.text_area(
                "Paste text here",
                height=200,
                placeholder=placeholder_text
            )
    
    if st.button("🔮 Predict Topic", type="primary", use_container_width=True):
        papers_to_process = []
        
        if uploaded_files and len(uploaded_files) > 0:
            with st.spinner(f"Extracting text from {len(uploaded_files)} PDF(s)..."):
                for uploaded_file in uploaded_files:
                    paper_text = extract_text_from_pdf(uploaded_file)
                    if paper_text:
                        paper_text = preprocess_text(paper_text)
                        papers_to_process.append((uploaded_file.name, paper_text))
                st.success(f"✅ Extracted text from {len(papers_to_process)} file(s)")
        elif manual_text:
            paper_text = preprocess_text(manual_text)
            papers_to_process.append(("Manual Input", paper_text))
        else:
            st.warning("⚠️ Please upload a PDF or paste text to continue.")
            return
        
        for paper_idx, (paper_name, paper_text) in enumerate(papers_to_process):
            if paper_text and len(paper_text) > 50:
                with st.spinner(f"Analyzing '{paper_name}' with {model_choice} on {dataset_choice}..."):
                    predicted_class, confidence, all_probs, top5_probs, top5_indices, graph_data, target_idx = predict_topic(
                        paper_text, selected_model, model_choice, data, dataset_config
                    )
                
                st.markdown("---")
                if len(papers_to_process) > 1:
                    st.markdown(f"## 🎯 Prediction Results for '{paper_name}'")
                else:
                    st.markdown("## 🎯 Prediction Results")
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Topic</h2>
                    <h1>{categories[predicted_class]}</h1>
                    <h3>Confidence: {confidence:.1%}</h3>
                    <p style="font-size:0.9em; opacity:0.9;">Dataset: {dataset_choice} | Model: {model_choice}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 🕸️ Knowledge Graph Visualization")
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    kg_fig = create_knowledge_graph_visualization(
                        graph_data, target_idx, categories, predicted_class, top5_indices, top5_probs, confidence
                    )
                    st.plotly_chart(kg_fig, use_container_width=True)
                
                with col_viz2:
                    dist_fig = create_topic_distribution_graph(all_probs, categories, top5_indices)
                    st.plotly_chart(dist_fig, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 📊 Top 5 Predictions")
                
                for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
                    idx_val = idx.item()
                    prob_val = prob.item()
                    
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{i+1}. {categories[idx_val]}**")
                        st.progress(prob_val)
                    with col_b:
                        st.metric("Probability", f"{prob_val:.1%}", label_visibility="collapsed")
                
                st.markdown("---")
                st.markdown("### 📝 Paper Preview")
                with st.expander("View extracted text (first 1000 characters)"):
                    st.text(paper_text[:1000] + "..." if len(paper_text) > 1000 else paper_text)
                
                st.markdown("---")
                predictions_df = pd.DataFrame({
                    'Rank': range(1, 6),
                    'Topic': [categories[idx.item()] for idx in top5_indices],
                    'Confidence': [f"{prob.item():.2%}" for prob in top5_probs]
                })
                
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=predictions_df.to_csv(index=False),
                    file_name=f"predictions_{paper_name.replace('.pdf', '')}_{dataset_choice.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"download_btn_{paper_idx}_{paper_name}"
                )
            else:
                st.error(f"⚠️ Text from '{paper_name}' is too short. Please provide more content.")
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Powered by PyTorch Geometric • Graph Neural Networks</p>
        <p>Models: GAT, GCN, GraphSAGE | Datasets: OGB arXiv (169K papers) & AMiner (10K authors)</p>
        <p>Current Dataset: <strong>{dataset_choice}</strong> | Current Model: <strong>{model_choice}</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
