"""
Research Compass - GNN Paper Classification System
Streamlit UI for predicting research paper topics using Graph Neural Networks
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from pathlib import Path
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import plotly.graph_objects as go
import networkx as nx
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Research Compass - GNN Paper Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 30px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model Definitions
class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCN(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(nn.Module):
    """GraphSAGE Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# arXiv CS category mapping (40 topics)
ARXIV_CATEGORIES = {
    0: "cs.AI - Artificial Intelligence",
    1: "cs.AR - Hardware Architecture",
    2: "cs.CC - Computational Complexity",
    3: "cs.CE - Computational Engineering",
    4: "cs.CG - Computational Geometry",
    5: "cs.CL - Computation and Language",
    6: "cs.CR - Cryptography and Security",
    7: "cs.CV - Computer Vision",
    8: "cs.CY - Computers and Society",
    9: "cs.DB - Databases",
    10: "cs.DC - Distributed Computing",
    11: "cs.DL - Digital Libraries",
    12: "cs.DM - Discrete Mathematics",
    13: "cs.DS - Data Structures and Algorithms",
    14: "cs.ET - Emerging Technologies",
    15: "cs.FL - Formal Languages",
    16: "cs.GL - General Literature",
    17: "cs.GR - Graphics",
    18: "cs.GT - Computer Science and Game Theory",
    19: "cs.HC - Human-Computer Interaction",
    20: "cs.IR - Information Retrieval",
    21: "cs.IT - Information Theory",
    22: "cs.LG - Machine Learning",
    23: "cs.LO - Logic in Computer Science",
    24: "cs.MA - Multiagent Systems",
    25: "cs.MM - Multimedia",
    26: "cs.MS - Mathematical Software",
    27: "cs.NA - Numerical Analysis",
    28: "cs.NE - Neural and Evolutionary Computing",
    29: "cs.NI - Networking and Internet Architecture",
    30: "cs.OH - Other Computer Science",
    31: "cs.OS - Operating Systems",
    32: "cs.PF - Performance",
    33: "cs.PL - Programming Languages",
    34: "cs.RO - Robotics",
    35: "cs.SC - Symbolic Computation",
    36: "cs.SD - Sound",
    37: "cs.SE - Software Engineering",
    38: "cs.SI - Social and Information Networks",
    39: "cs.SY - Systems and Control"
}

# AMiner research field mapping (10 topics)
AMINER_CATEGORIES = {
    0: "Machine Learning & AI",
    1: "Data Mining & Analytics",
    2: "Computer Vision & Graphics",
    3: "Natural Language Processing",
    4: "Databases & Information Systems",
    5: "Networks & Distributed Systems",
    6: "Software Engineering",
    7: "Security & Cryptography",
    8: "Theory & Algorithms",
    9: "Systems & Architecture"
}

DATASET_INFO = {
    "OGB arXiv": {
        "categories": ARXIV_CATEGORIES,
        "model_file": "saved_models/all_models.pt",
        "num_classes": 40,
        "feature_dim": 128,
        "description": "169K CS papers from arXiv (1993-2020)",
        "task": "Paper topic classification"
    },
    "AMiner": {
        "categories": AMINER_CATEGORIES,
        "model_file": "saved_models/aminer_models.pt",
        "num_classes": 10,
        "feature_dim": 503,  # Based on AMiner dataset
        "description": "10K authors from AMiner network",
        "task": "Research field prediction"
    }
}

@st.cache_resource
def load_models(dataset_name="OGB arXiv"):
    """Load trained GNN models for the selected dataset"""
    device = torch.device('cpu')
    dataset_config = DATASET_INFO[dataset_name]
    model_path = Path(dataset_config['model_file'])
    
    if not model_path.exists():
        return None, None, None, None, dataset_config
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get feature dimension and num classes
    feature_dim = dataset_config['feature_dim']
    num_classes = dataset_config['num_classes']
    
    # Initialize models
    gat = GAT(feature_dim, 128, num_classes)
    gcn = GCN(feature_dim, 128, num_classes)
    sage = GraphSAGE(feature_dim, 128, num_classes)
    
    # Load weights
    gat.load_state_dict(checkpoint['gat_node'])
    gcn.load_state_dict(checkpoint['gcn_node'])
    sage.load_state_dict(checkpoint['sage_node'])
    
    gat.eval()
    gcn.eval()
    sage.eval()
    
    results_node = pd.DataFrame(checkpoint['results_node'])
    
    return gat, gcn, sage, results_node, dataset_config

@st.cache_resource
def load_dataset_for_features(dataset_name="OGB arXiv"):
    """Load dataset to get feature statistics"""
    dataset_config = DATASET_INFO[dataset_name]
    
    if dataset_name == "OGB arXiv":
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            import torch.serialization
            from torch_geometric.data import Data
            from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
            from torch_geometric.data.storage import GlobalStorage, EdgeStorage, NodeStorage
            
            torch.serialization.add_safe_globals([
                DataEdgeAttr, DataTensorAttr, Data, 
                GlobalStorage, EdgeStorage, NodeStorage
            ])
            
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/')
            data = dataset[0]
            return data
        except Exception as e:
            st.error(f"Error loading OGB arXiv dataset: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    elif dataset_name == "AMiner":
        # Create a dummy dataset structure for AMiner
        try:
            import torch
            import numpy as np
            from torch_geometric.data import Data
            import torch.nn.functional as F
            
            # Load or create minimal graph for context
            num_nodes = 1000
            feature_dim = dataset_config['feature_dim']
            
            # Random features for context
            x = torch.randn(num_nodes, feature_dim)
            x = F.normalize(x, p=2, dim=1)
            
            # Create some edges
            edge_list = []
            for i in range(num_nodes):
                for _ in range(5):
                    j = np.random.randint(0, num_nodes)
                    if i != j:
                        edge_list.append([i, j])
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            data = Data(x=x, edge_index=edge_index)
            return data
        except Exception as e:
            st.error(f"Error creating AMiner dataset: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    return None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def preprocess_text(text):
    """Clean and preprocess extracted text"""
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text

def text_to_features(text, target_dim=128):
    """Convert text to feature vector matching the model input"""
    # Use TF-IDF to create a feature vector
    vectorizer = TfidfVectorizer(max_features=target_dim, stop_words='english')
    
    # Fit and transform on the input text
    try:
        vector = vectorizer.fit_transform([text]).toarray()[0]
        # Pad or truncate to exact target_dim
        if len(vector) < target_dim:
            vector = np.pad(vector, (0, target_dim - len(vector)))
        elif len(vector) > target_dim:
            vector = vector[:target_dim]
        # Normalize
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        return torch.FloatTensor(vector)
    except:
        # Return random normalized features if text is too short
        vector = np.random.randn(target_dim)
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        return torch.FloatTensor(vector)

def create_dummy_graph(features, data):
    """Create a minimal graph structure for prediction"""
    # Use a small subset of the original graph for context
    n_context = min(100, data.num_nodes)
    
    # Get features from original data as context
    context_features = data.x[:n_context]
    
    # Add the new paper as a node
    new_features = torch.cat([context_features, features.unsqueeze(0)], dim=0)
    
    # Create edges: connect new node to a few context nodes
    new_node_idx = n_context
    edge_sources = [new_node_idx] * 5 + list(range(5))
    edge_targets = list(range(5)) + [new_node_idx] * 5
    
    # Get some edges from original graph
    original_edges = data.edge_index[:, :500]
    original_edges = original_edges[:, original_edges[0] < n_context]
    original_edges = original_edges[:, original_edges[1] < n_context]
    
    # Combine edges
    new_edges = torch.LongTensor([edge_sources, edge_targets])
    edge_index = torch.cat([original_edges, new_edges], dim=1)
    
    return Data(x=new_features, edge_index=edge_index), new_node_idx

def predict_topic(text, model, model_name, data, dataset_config):
    """Predict paper topic using the selected model"""
    # Convert text to features
    features = text_to_features(text, target_dim=dataset_config['feature_dim'])
    
    # Create graph
    graph_data, target_idx = create_dummy_graph(features, data)
    
    # Predict
    with torch.no_grad():
        output = model(graph_data.x, graph_data.edge_index)
        probabilities = F.softmax(output[target_idx], dim=0)
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()
    
    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probabilities, k=min(5, len(probabilities)))
    
    return predicted_class, confidence, probabilities, top5_probs, top5_indices, graph_data, target_idx

def create_knowledge_graph_visualization(graph_data, target_idx, categories, predicted_class, top5_indices, top5_probs, confidence):
    """Create interactive knowledge graph visualization using Plotly"""
    
    # Limit to small subgraph for visualization
    num_nodes = min(50, graph_data.x.shape[0])
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(num_nodes):
        node_type = "target" if i == target_idx else "context"
        G.add_node(i, node_type=node_type)
    
    # Add edges (limit to nodes in our subset)
    edge_index = graph_data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < num_nodes and dst < num_nodes:
            G.add_edge(int(src), int(dst))
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge trace with labels
    edge_x = []
    edge_y = []
    edge_mid_x = []
    edge_mid_y = []
    edge_labels = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Calculate midpoint for label
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        edge_mid_x.append(mid_x)
        edge_mid_y.append(mid_y)
        edge_labels.append(f"{edge[0]}→{edge[1]}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#555'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Edge labels trace
    edge_label_trace = go.Scatter(
        x=edge_mid_x, y=edge_mid_y,
        mode='text',
        text=edge_labels,
        textfont=dict(size=8, color='#444'),
        hoverinfo='none',
        showlegend=False
    )
    
    # Create node traces with labels showing categories
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
            # Shorten category name for display
            short_name = category_name.split('.')[-1][:15] if '.' in category_name else category_name[:15]
            node_text.append(f"📄 Your Paper<br>Predicted: {category_name}<br>Confidence: {confidence:.1%}")
            node_labels.append(f"YOU\n{short_name}")
        elif node < 5:  # Similar papers
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
            # Random category for context nodes (just for visualization)
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
            line=dict(width=2.5, color='#333')
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text='🕸️ Knowledge Graph: Paper Citation Network',
                           x=0.5,
                           xanchor='center'
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='rgba(240,240,240,0.5)',
                       height=500
                   ))
    
    return fig

def create_topic_distribution_graph(probabilities, categories, top5_indices):
    """Create an interactive topic distribution visualization"""
    
    # Get all probabilities
    all_probs = probabilities.numpy()
    top_k = min(15, len(all_probs))
    top_indices = np.argsort(all_probs)[-top_k:][::-1]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[categories[i] for i in top_indices],
            y=[all_probs[i] * 100 for i in top_indices],
            marker=dict(
                color=[all_probs[i] for i in top_indices],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence")
            ),
            text=[f"{all_probs[i]:.1%}" for i in top_indices],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='📊 Topic Probability Distribution',
        xaxis_title='Research Topics',
        yaxis_title='Confidence (%)',
        xaxis={'tickangle': -45},
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">🎓 Research Compass</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Research Paper Topic Classification</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
        st.title("⚙️ Configuration")
        
        # Dataset selection
        dataset_choice = st.selectbox(
            "📊 Select Dataset",
            ["OGB arXiv", "AMiner"],
            help="Choose the dataset for prediction"
        )
        
        model_choice = st.selectbox(
            "🤖 Select GNN Model",
            ["GAT (Graph Attention)", "GCN (Graph Convolution)", "GraphSAGE"],
            help="Choose the Graph Neural Network architecture for prediction"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        
        dataset_info = DATASET_INFO[dataset_choice]
        st.info(f"""
        **{dataset_choice}**
        - {dataset_info['description']}
        - **Task**: {dataset_info['task']}
        - **Topics**: {dataset_info['num_classes']} categories
        - **Features**: {dataset_info['feature_dim']} dimensions
        """)
        
        st.markdown("---")
        st.markdown("### 🧠 Model Info")
        st.info("""
        **Models Available:**
        - **GAT**: Uses attention mechanism to weight neighbor importance
        - **GCN**: Classic graph convolution approach
        - **GraphSAGE**: Samples and aggregates features from neighbors
        """)
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Select a dataset
        2. Upload a research paper (PDF) or paste text
        3. Choose a GNN model
        4. Click 'Predict Topic'
        5. View predictions and confidence scores
        """)
    
    # Load models
    with st.spinner(f"Loading {dataset_choice} models..."):
        gat, gcn, sage, results, dataset_config = load_models(dataset_choice)
        data = load_dataset_for_features(dataset_choice)
    
    if gat is None or data is None:
        st.error(f"⚠️ Models or dataset not found for {dataset_choice}. Please train the models first by running the notebook.")
        if dataset_choice == "OGB arXiv":
            st.info("Run all cells in `GNN_Complete_Standalone.ipynb` to train and save the models.")
        else:
            st.info("Run all cells in `GNN_AMiner_Real_Fixed.ipynb` to train and save the models.")
        return
    
    # Map model choice
    model_map = {
        "GAT (Graph Attention)": gat,
        "GCN (Graph Convolution)": gcn,
        "GraphSAGE": sage
    }
    selected_model = model_map[model_choice]
    
    # Get category mapping for current dataset
    categories = dataset_config['categories']
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 🏆 Model Performance")
        if results is not None:
            for idx, row in results.iterrows():
                st.metric(
                    label=row['Model'],
                    value=f"{row['Test Accuracy']:.2%}",
                    delta=None
                )
        st.markdown("---")
    
    with col1:
        st.markdown("### 📄 Upload Research Paper")
        
        # File uploader (multiple files)
        uploaded_files = st.file_uploader(
            "Choose PDF file(s)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more research papers in PDF format"
        )
        
        # Text input alternative
        with st.expander("Or paste paper abstract/text"):
            placeholder_text = "Enter the abstract or full text of your research paper..."
            if dataset_choice == "AMiner":
                placeholder_text = "Enter author information or research topics..."
            manual_text = st.text_area(
                "Paste text here",
                height=200,
                placeholder=placeholder_text
            )
    
    # Prediction button
    if st.button("🔮 Predict Topic", type="primary", use_container_width=True):
        papers_to_process = []
        
        # Get text from files or manual input
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
        
        # Process each paper
        for paper_idx, (paper_name, paper_text) in enumerate(papers_to_process):
            if paper_text and len(paper_text) > 50:
                # Make prediction
                with st.spinner(f"Analyzing '{paper_name}' with {model_choice} on {dataset_choice}..."):
                    predicted_class, confidence, all_probs, top5_probs, top5_indices, graph_data, target_idx = predict_topic(
                        paper_text, selected_model, model_choice, data, dataset_config
                    )
                
                # Display results
                st.markdown("---")
                if len(papers_to_process) > 1:
                    st.markdown(f"## 🎯 Prediction Results for '{paper_name}'")
                else:
                    st.markdown("## 🎯 Prediction Results")
                
                # Main prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Topic</h2>
                    <h1>{categories[predicted_class]}</h1>
                    <h3>Confidence: {confidence:.1%}</h3>
                    <p style="font-size:0.9em; opacity:0.9;">Dataset: {dataset_choice} | Model: {model_choice}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Knowledge Graph Visualization
                st.markdown("---")
                st.markdown("### 🕸️ Knowledge Graph Visualization")
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    # Citation network graph
                    kg_fig = create_knowledge_graph_visualization(
                        graph_data, target_idx, categories, predicted_class, top5_indices, top5_probs, confidence
                    )
                    st.plotly_chart(kg_fig, use_container_width=True)
                
                with col_viz2:
                    # Topic distribution
                    dist_fig = create_topic_distribution_graph(all_probs, categories, top5_indices)
                    st.plotly_chart(dist_fig, use_container_width=True)
                
                # Top 5 predictions
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
                        st.metric("", f"{prob_val:.1%}")
                
                # Show paper preview
                st.markdown("---")
                st.markdown("### 📝 Paper Preview")
                with st.expander("View extracted text (first 1000 characters)"):
                    st.text(paper_text[:1000] + "..." if len(paper_text) > 1000 else paper_text)
                
                # Download predictions
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
    
    # Footer
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
