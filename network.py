"""
Network topology and node simulation module for quantum-classical hybrid networks.
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, Set
import os


class HybridNetworkSimulator:
    """Simulates a hybrid quantum-classical network topology."""
    
    def __init__(self, num_nodes: int = 10, quantum_node_ratio: float = 0.4, 
                 prob_edge: float = 0.3, distance_range: tuple = (10, 100)):
        """Initialize the hybrid quantum-classical network."""
        self.num_nodes = num_nodes
        self.quantum_node_ratio = quantum_node_ratio
        self.prob_edge = prob_edge
        self.distance_range = distance_range
        
        self.G = nx.Graph()
        self.quantum_nodes: Set[int] = set()
        self.classical_nodes: Set[int] = set()
        self.node_positions = {}
        self.link_properties = {}
        
        # Create network
        self._create_nodes()
        self._create_edges()
        self._initialize_link_properties()
        
    def _create_nodes(self):
        """Create quantum and classical nodes."""
        num_quantum = int(self.num_nodes * self.quantum_node_ratio)
        
        # Randomly select quantum nodes
        quantum_indices = random.sample(range(self.num_nodes), num_quantum)
        
        for i in range(self.num_nodes):
            if i in quantum_indices:
                self.G.add_node(i, type='quantum', entanglement_capable=True)
                self.quantum_nodes.add(i)
            else:
                self.G.add_node(i, type='classical', entanglement_capable=False)
                self.classical_nodes.add(i)
    
    def _create_edges(self):
        """Create edges ensuring connected network."""
        # Use Erdős-Rényi model for initial connectivity
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if random.random() < self.prob_edge:
                    self.G.add_edge(i, j)
        
        # Ensure the graph is connected
        if not nx.is_connected(self.G):
            components = list(nx.connected_components(self.G))
            for i in range(1, len(components)):
                # Connect each component to the main component
                node1 = random.choice(list(components[0]))
                node2 = random.choice(list(components[i]))
                self.G.add_edge(node1, node2)
    
    def _initialize_link_properties(self):
        """Initialize properties for each link."""
        for edge in self.G.edges():
            node1, node2 = edge
            distance = random.uniform(*self.distance_range)  # km
            
            # Determine link type
            if (self.G.nodes[node1]['type'] == 'quantum' and 
                self.G.nodes[node2]['type'] == 'quantum'):
                link_type = 'quantum'
                base_success_rate = 0.85
            else:
                link_type = 'classical'
                base_success_rate = 0.95
            
            self.link_properties[edge] = {
                'type': link_type,
                'distance': distance,
                'base_success_rate': base_success_rate
            }
    
    def visualize_network(self, save_path: str = None):
        """Visualize the hybrid network."""
        plt.figure(figsize=(12, 8))
        
        # Position nodes using spring layout
        self.node_positions = nx.spring_layout(self.G, k=2, iterations=50)
        
        # Draw quantum nodes
        nx.draw_networkx_nodes(self.G, self.node_positions, 
                              nodelist=list(self.quantum_nodes),
                              node_color='red', node_size=700,
                              node_shape='s', label='Quantum Nodes')
        
        # Draw classical nodes
        nx.draw_networkx_nodes(self.G, self.node_positions,
                              nodelist=list(self.classical_nodes),
                              node_color='blue', node_size=700,
                              node_shape='o', label='Classical Nodes')
        
        # Draw edges
        quantum_edges = [(u, v) for u, v in self.G.edges() 
                        if self.link_properties[(u, v)]['type'] == 'quantum']
        classical_edges = [(u, v) for u, v in self.G.edges() 
                          if self.link_properties[(u, v)]['type'] == 'classical']
        
        nx.draw_networkx_edges(self.G, self.node_positions, 
                              edgelist=quantum_edges,
                              edge_color='red', style='dashed', width=2)
        nx.draw_networkx_edges(self.G, self.node_positions,
                              edgelist=classical_edges,
                              edge_color='blue', width=1)
        
        # Draw labels
        nx.draw_networkx_labels(self.G, self.node_positions)
        
        plt.title(f"Hybrid Quantum-Classical Network ({self.num_nodes} nodes)")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network topology saved to {save_path}")
        
        plt.show()
    
    def get_network_stats(self) -> Dict:
        """Get basic network statistics."""
        return {
            'total_nodes': self.num_nodes,
            'quantum_nodes': len(self.quantum_nodes),
            'classical_nodes': len(self.classical_nodes),
            'total_edges': self.G.number_of_edges(),
            'quantum_edges': len([(u, v) for u, v in self.G.edges() 
                                 if self.link_properties[(u, v)]['type'] == 'quantum']),
            'classical_edges': len([(u, v) for u, v in self.G.edges() 
                                   if self.link_properties[(u, v)]['type'] == 'classical']),
            'is_connected': nx.is_connected(self.G),
            'average_degree': sum(dict(self.G.degree()).values()) / self.num_nodes,
            'diameter': nx.diameter(self.G) if nx.is_connected(self.G) else None
        }
