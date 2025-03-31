import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple

class FeatureMapper:
    def __init__(self):
        """Initialize FeatureMapper with mapping graph."""
        self.mapping_graph = nx.Graph()
        self.feature_metadata = {}
        
    def add_feature_mapping(self, source1: str, feature1: str, 
                          source2: str, feature2: str, 
                          confidence: float = 1.0):
        """Add a mapping between two features from different sources."""
        node1 = f"{source1}:{feature1}"
        node2 = f"{source2}:{feature2}"
        
        self.mapping_graph.add_edge(node1, node2, weight=confidence)
        
    def get_equivalent_features(self, source: str, feature: str) -> List[Tuple[str, str]]:
        """Get all equivalent features across databases."""
        node = f"{source}:{feature}"
        if node not in self.mapping_graph:
            return []
            
        equivalent = []
        for neighbor in self.mapping_graph.neighbors(node):
            src, feat = neighbor.split(":", 1)
            equivalent.append((src, feat))
        return equivalent
        
    def visualize_mappings(self, output_file: str = None):
        """Create a visualization of feature mappings."""
        import plotly.graph_objects as go
        
        # Create plotly network graph
        pos = nx.spring_layout(self.mapping_graph)
        
        edge_x = []
        edge_y = []
        for edge in self.mapping_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        node_x = [pos[node][0] for node in self.mapping_graph.nodes()]
        node_y = [pos[node][1] for node in self.mapping_graph.nodes()]
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines'))
        
        # Add nodes
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                               text=list(self.mapping_graph.nodes()),
                               textposition="bottom center"))
        
        if output_file:
            fig.write_html(output_file)
        return fig
