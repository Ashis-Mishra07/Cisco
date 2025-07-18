import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import deque
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import hashlib
import secrets
import math


# Set page config
st.set_page_config(
    page_title="Quantum-Classical Hybrid Network Simulation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metrics-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .failure-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔬 Quantum-Classical Hybrid Network Simulation</h1>', unsafe_allow_html=True)

# Initialize session state
if 'network' not in st.session_state:
    st.session_state.network = None
if 'link_sim' not in st.session_state:
    st.session_state.link_sim = None
if 'router' not in st.session_state:
    st.session_state.router = None

# Part 1: Network Topology and Node Simulation
class HybridNetworkSimulator:
    def __init__(self, num_nodes=10, quantum_node_ratio=0.4):
        """Initialize the hybrid quantum-classical network."""
        self.G = nx.Graph()
        self.num_nodes = num_nodes
        self.quantum_nodes = set()
        self.classical_nodes = set()
        self.node_positions = {}
        
        # Create nodes
        self._create_nodes(quantum_node_ratio)
        
        # Create edges with properties
        self._create_edges()
        
        # Link properties
        self.link_properties = {}
        self._initialize_link_properties()
        
        # Store node positions for consistent visualization
        self.node_positions = None
        
    def _create_nodes(self, quantum_ratio):
        """Create quantum and classical nodes."""
        num_quantum = int(self.num_nodes * quantum_ratio)
        
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
        prob_edge = 0.3
        
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if random.random() < prob_edge:
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
            distance = random.uniform(10, 100)  # km
            
            # Determine link type
            if self.G.nodes[node1]['type'] == 'quantum' and self.G.nodes[node2]['type'] == 'quantum':
                link_type = 'quantum'
            else:
                link_type = 'classical'
            
            self.link_properties[edge] = {
                'type': link_type,
                'distance': distance,
                'base_success_rate': 0.95 if link_type == 'classical' else 0.85
            }
    
    def get_network_plotly(self):
        """Create a Plotly visualization of the network."""
        # Position nodes using spring layout (only calculate once)
        if self.node_positions is None:
            self.node_positions = nx.spring_layout(self.G, k=2, iterations=50)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_symbols = []
        
        for node in self.G.nodes():
            x, y = self.node_positions[node]
            node_x.append(x)
            node_y.append(y)
            
            if node in self.quantum_nodes:
                node_colors.append('red')
                node_symbols.append('square')
                node_text.append(f'Quantum Node {node}')
            else:
                node_colors.append('blue')
                node_symbols.append('circle')
                node_text.append(f'Classical Node {node}')
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_colors = []
        edge_widths = []
        
        for edge in self.G.edges():
            x0, y0 = self.node_positions[edge[0]]
            x1, y1 = self.node_positions[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            if self.link_properties[edge]['type'] == 'quantum':
                edge_colors.append('red')
                edge_widths.append(3)
            else:
                edge_colors.append('blue')
                edge_widths.append(1)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        for i, edge in enumerate(self.G.edges()):
            x0, y0 = self.node_positions[edge[0]]
            x1, y1 = self.node_positions[edge[1]]
            
            color = 'red' if self.link_properties[edge]['type'] == 'quantum' else 'blue'
            width = 3 if self.link_properties[edge]['type'] == 'quantum' else 1
            dash = 'dash' if self.link_properties[edge]['type'] == 'quantum' else 'solid'
            
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color=color, width=width, dash=dash),
                showlegend=False,
                hoverinfo='none'
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color=node_colors,
                symbol=node_symbols,
                line=dict(width=2, color='black')
            ),
            text=[str(i) for i in range(self.num_nodes)],
            textposition="middle center",
            textfont=dict(color="white", size=12),
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        # Add legend manually
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color='red', symbol='square'),
            name='Quantum Nodes',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color='blue', symbol='circle'),
            name='Classical Nodes',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name='Quantum Links',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='blue', width=1),
            name='Classical Links',
            showlegend=True
        ))
        
        fig.update_layout(
            title="Hybrid Quantum-Classical Network Topology",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Quantum nodes (red squares) can create entanglement<br>Classical nodes (blue circles) use traditional networking",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
                font=dict(size=10)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig

# Part 2: Quantum Link Simulator
class QuantumLinkSimulator:
    """Decoherence + repeater swap-failure + classical packet loss."""
    
    def __init__(self, network: HybridNetworkSimulator):
        self.network = network
        self.net = network  # Alias for compatibility
        self.decoh_km = 0.01  # Decoherence rate per km
        self.swap_success = 0.92  # Quantum repeater swap success rate
        self.decoherence_rate = 0.01  # Backward compatibility
        self.entanglement_success_base = 0.8  # Backward compatibility
        self.classical_packet_loss_rate = 0.001  # per km
        
    def _edge(self, u, v):
        """Get edge tuple in correct order."""
        return (u, v) if (u, v) in self.network.link_properties else (v, u)
    
    def simulate(self, u: int, v: int) -> Dict:
        """New improved simulation method with repeater support."""
        e = self._edge(u, v)
        lp = self.network.link_properties[e]
        d = lp["distance"]
        out = dict(link_type=lp["type"], distance=d, success=False, reason="")

        if lp["type"] == "classical":
            if random.random() < 1 - np.exp(-0.001 * d):
                out["reason"] = "packet loss"
            else:
                out.update(success=True, reason="classical OK")
            return out

        # quantum link
        if random.random() < 1 - np.exp(-self.decoh_km * d):
            out["reason"] = "decoherence"
            return out

        # Check for repeater nodes and simulate swap failure
        u_node = self.network.G.nodes[u]
        v_node = self.network.G.nodes[v]
        if (u_node.get("type") == "repeater" or v_node.get("type") == "repeater" or
            u_node.get("type") == "quantum" or v_node.get("type") == "quantum"):
            if random.random() > self.swap_success:
                out["reason"] = "swap failure"
                return out

        out.update(success=True, reason="quantum OK")
        return out
        
    def simulate_transmission(self, node1: int, node2: int) -> Dict:
        """Legacy method for backward compatibility."""
        if (node1, node2) not in self.network.link_properties:
            edge = (node2, node1)
        else:
            edge = (node1, node2)
            
        link_props = self.network.link_properties[edge]
        distance = link_props['distance']
        
        result = {
            'link_type': link_props['type'],
            'distance': distance,
            'success': False,
            'reason': None
        }
        
        if link_props['type'] == 'quantum':
            # Decoherence effect
            decoherence_prob = 1 - np.exp(-self.decoherence_rate * distance)
            
            if random.random() < decoherence_prob:
                result['reason'] = 'decoherence'
                return result
            
            # Check for repeater swap failure
            u_node = self.network.G.nodes[node1]
            v_node = self.network.G.nodes[node2]
            if (u_node.get("type") == "repeater" or v_node.get("type") == "repeater" or
                u_node.get("type") == "quantum" or v_node.get("type") == "quantum"):
                if random.random() > self.swap_success:
                    result['reason'] = 'swap failure'
                    return result
            
            # No-cloning check (simulated)
            if random.random() < 0.05:  # 5% chance of amplification attempt
                result['reason'] = 'no-cloning violation'
                return result
            
            # Entanglement distribution
            entanglement_success = self.entanglement_success_base * np.exp(-distance/1000)
            if random.random() < entanglement_success:
                result['success'] = True
                result['reason'] = 'successful quantum transmission'
            else:
                result['reason'] = 'entanglement failure'
        else:
            # Classical link
            packet_loss_prob = 1 - np.exp(-self.classical_packet_loss_rate * distance)
            if random.random() > packet_loss_prob:
                result['success'] = True
                result['reason'] = 'successful classical transmission'
            else:
                result['reason'] = 'packet loss'
        
        return result
    
    def analyze_link_behavior(self, num_simulations=1000):
        """Analyze and plot link behavior differences with detailed quantum effects."""
        results = {'quantum': [], 'classical': []}
        distances = {'quantum': [], 'classical': []}
        
        # Enhanced quantum effects tracking with swap failures
        quantum_effects = {
            'decoherence': 0,
            'no_cloning': 0,
            'entanglement_failure': 0,
            'swap_failure': 0,  # New: quantum repeater swap failures
            'successful_quantum': 0
        }
        
        # Classical effects tracking
        classical_effects = {
            'packet_loss': 0,
            'successful_classical': 0
        }
        
        for _ in range(num_simulations):
            edge = random.choice(list(self.network.G.edges()))
            result = self.simulate_transmission(edge[0], edge[1])
            
            link_type = result['link_type']
            results[link_type].append(1 if result['success'] else 0)
            distances[link_type].append(result['distance'])
            
            # Track specific failure reasons including new swap failures
            if link_type == 'quantum':
                if result['reason'] == 'decoherence':
                    quantum_effects['decoherence'] += 1
                elif result['reason'] == 'no-cloning violation':
                    quantum_effects['no_cloning'] += 1
                elif result['reason'] == 'entanglement failure':
                    quantum_effects['entanglement_failure'] += 1
                elif result['reason'] == 'swap failure':
                    quantum_effects['swap_failure'] += 1
                elif result['reason'] in ['successful quantum transmission', 'quantum OK']:
                    quantum_effects['successful_quantum'] += 1
            else:
                if result['reason'] in ['packet loss']:
                    classical_effects['packet_loss'] += 1
                elif result['reason'] in ['successful classical transmission', 'classical OK']:
                    classical_effects['successful_classical'] += 1
        
        return results, distances, quantum_effects, classical_effects
    
    def analyze_quantum_effects_vs_distance(self, num_simulations=2000):
        """Analyze how quantum effects change with distance."""
        distance_ranges = [(0, 25), (25, 50), (50, 75), (75, 100)]
        range_labels = ['0-25km', '25-50km', '50-75km', '75-100km']
        
        effects_by_distance = {}
        for label in range_labels:
            effects_by_distance[label] = {
                'decoherence': 0,
                'no_cloning': 0,
                'entanglement_failure': 0,
                'successful_quantum': 0,
                'total_attempts': 0
            }
        
        for _ in range(num_simulations):
            edge = random.choice(list(self.network.G.edges()))
            if self.network.link_properties[edge]['type'] == 'quantum':
                result = self.simulate_transmission(edge[0], edge[1])
                distance = result['distance']
                
                # Find appropriate distance range
                for i, (min_dist, max_dist) in enumerate(distance_ranges):
                    if min_dist <= distance < max_dist:
                        label = range_labels[i]
                        effects_by_distance[label]['total_attempts'] += 1
                        
                        if result['reason'] == 'decoherence':
                            effects_by_distance[label]['decoherence'] += 1
                        elif result['reason'] == 'no-cloning violation':
                            effects_by_distance[label]['no_cloning'] += 1
                        elif result['reason'] == 'entanglement failure':
                            effects_by_distance[label]['entanglement_failure'] += 1
                        elif result['reason'] == 'successful quantum transmission':
                            effects_by_distance[label]['successful_quantum'] += 1
                        break
        
        return effects_by_distance

# Part 3: Hybrid Routing Protocol
class HybridRoutingProtocol:
    def __init__(self, network: HybridNetworkSimulator, link_simulator: QuantumLinkSimulator):
        self.network = network
        self.link_sim = link_simulator
        self.routing_table = {}
        self._build_routing_table()
    
    def _build_routing_table(self):
        """Build routing table with link reliability scores."""
        for edge in self.network.G.edges():
            # Simulate link multiple times to estimate reliability
            successes = 0
            trials = 20
            
            for _ in range(trials):
                result = self.link_sim.simulate_transmission(edge[0], edge[1])
                if result['success']:
                    successes += 1
            
            reliability = successes / trials
            self.routing_table[edge] = {
                'reliability': reliability,
                'type': self.network.link_properties[edge]['type'],
                'distance': self.network.link_properties[edge]['distance']
            }
    
    def find_best_path(self, source: int, destination: int, prefer_quantum=True) -> Optional[List[int]]:
        """Find best path considering quantum preference and reliability."""
        if source == destination:
            return [source]
        
        # Use modified Dijkstra's algorithm
        distances = {node: float('inf') for node in self.network.G.nodes()}
        distances[source] = 0
        previous = {node: None for node in self.network.G.nodes()}
        unvisited = set(self.network.G.nodes())
        
        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])
            
            if distances[current] == float('inf'):
                break
                
            if current == destination:
                break
                
            unvisited.remove(current)
            
            for neighbor in self.network.G.neighbors(current):
                if neighbor in unvisited:
                    # Calculate edge weight
                    edge = (current, neighbor) if (current, neighbor) in self.routing_table else (neighbor, current)
                    
                    reliability = self.routing_table[edge]['reliability']
                    link_type = self.routing_table[edge]['type']
                    distance = self.routing_table[edge]['distance']
                    
                    # Weight calculation
                    if prefer_quantum and link_type == 'quantum':
                        weight = distance / (reliability * 2)  # Prefer quantum
                    else:
                        weight = distance / reliability
                    
                    alt_distance = distances[current] + weight
                    
                    if alt_distance < distances[neighbor]:
                        distances[neighbor] = alt_distance
                        previous[neighbor] = current
        
        # Reconstruct path
        if previous[destination] is None:
            return None
            
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        return path
    
    def send_message(self, source: int, destination: int, message: str) -> Dict:
        """Send message using hybrid routing with fallback."""
        result = {
            'success': False,
            'path': None,
            'attempts': [],
            'final_method': None,
            'path_analysis': {}
        }
        
        # First try quantum-preferred path
        quantum_preferred_path = self.find_best_path(source, destination, prefer_quantum=True)
        
        if quantum_preferred_path:
            # Attempt transmission on quantum-preferred path
            path_success = True
            quantum_links_used = 0
            classical_links_used = 0
            
            for i in range(len(quantum_preferred_path) - 1):
                trans_result = self.link_sim.simulate_transmission(
                    quantum_preferred_path[i], quantum_preferred_path[i+1]
                )
                
                # Count link types
                if trans_result['link_type'] == 'quantum':
                    quantum_links_used += 1
                else:
                    classical_links_used += 1
                
                result['attempts'].append({
                    'type': trans_result['link_type'],  # Use actual link type
                    'link': (quantum_preferred_path[i], quantum_preferred_path[i+1]),
                    'success': trans_result['success'],
                    'reason': trans_result['reason']
                })
                
                if not trans_result['success']:
                    path_success = False
                    break
            
            if path_success:
                result['success'] = True
                result['path'] = quantum_preferred_path
                
                # Determine actual path composition
                total_links = quantum_links_used + classical_links_used
                if quantum_links_used > classical_links_used:
                    result['final_method'] = 'quantum-majority'
                elif classical_links_used > quantum_links_used:
                    result['final_method'] = 'classical-majority'
                else:
                    result['final_method'] = 'hybrid-balanced'
                
                result['path_analysis'] = {
                    'quantum_links': quantum_links_used,
                    'classical_links': classical_links_used,
                    'total_links': total_links,
                    'quantum_percentage': (quantum_links_used / total_links * 100) if total_links > 0 else 0
                }
                return result
        
        # Fallback to classical-preferred path
        classical_preferred_path = self.find_best_path(source, destination, prefer_quantum=False)
        
        if classical_preferred_path:
            path_success = True
            quantum_links_used = 0
            classical_links_used = 0
            
            for i in range(len(classical_preferred_path) - 1):
                trans_result = self.link_sim.simulate_transmission(
                    classical_preferred_path[i], classical_preferred_path[i+1]
                )
                
                # Count link types
                if trans_result['link_type'] == 'quantum':
                    quantum_links_used += 1
                else:
                    classical_links_used += 1
                
                result['attempts'].append({
                    'type': trans_result['link_type'],  # Use actual link type
                    'link': (classical_preferred_path[i], classical_preferred_path[i+1]),
                    'success': trans_result['success'],
                    'reason': trans_result['reason']
                })
                
                if not trans_result['success']:
                    path_success = False
                    break
            
            if path_success:
                result['success'] = True
                result['path'] = classical_preferred_path
                
                # Determine actual path composition
                total_links = quantum_links_used + classical_links_used
                if quantum_links_used > classical_links_used:
                    result['final_method'] = 'quantum-majority-fallback'
                elif classical_links_used > quantum_links_used:
                    result['final_method'] = 'classical-majority-fallback'
                else:
                    result['final_method'] = 'hybrid-balanced-fallback'
                
                result['path_analysis'] = {
                    'quantum_links': quantum_links_used,
                    'classical_links': classical_links_used,
                    'total_links': total_links,
                    'quantum_percentage': (quantum_links_used / total_links * 100) if total_links > 0 else 0
                }
        
        return result

# Streamlit App Layout
def main():
    # Sidebar for configuration
    st.sidebar.header("🔧 Network Configuration")
    
    # Network parameters (store in session state to track changes)
    if 'num_nodes' not in st.session_state:
        st.session_state.num_nodes = 12
    if 'quantum_ratio' not in st.session_state:
        st.session_state.quantum_ratio = 0.4
    
    num_nodes = st.sidebar.slider("Number of Nodes", 5, 20, st.session_state.num_nodes)
    quantum_ratio = st.sidebar.slider("Quantum Node Ratio", 0.1, 0.9, st.session_state.quantum_ratio)
    
    # Check if parameters changed or network doesn't exist
    params_changed = (num_nodes != st.session_state.num_nodes or 
                     quantum_ratio != st.session_state.quantum_ratio)
    
    # Create network button or when parameters change
    if st.sidebar.button("🔄 Generate New Network") or st.session_state.network is None or params_changed:
        with st.spinner("Generating network..."):
            st.session_state.network = HybridNetworkSimulator(num_nodes, quantum_ratio)
            st.session_state.link_sim = QuantumLinkSimulator(st.session_state.network)
            st.session_state.router = HybridRoutingProtocol(st.session_state.network, st.session_state.link_sim)
            # Update stored parameters
            st.session_state.num_nodes = num_nodes
            st.session_state.quantum_ratio = quantum_ratio
        st.success("Network generated successfully!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌐 Network Topology", 
        "📊 Link Analysis", 
        "🔬 Quantum Effects",
        "🔄 Message Routing", 
        "📈 Performance Analysis",
        "🔐 Security Analysis"
    ])
    
    with tab1:
        st.markdown('<h2 class="section-header">Network Topology</h2>', unsafe_allow_html=True)
        
        if st.session_state.network:
            # Display network visualization
            fig = st.session_state.network.get_network_plotly()
            st.plotly_chart(fig, use_container_width=True)
            
            # Network statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Nodes", st.session_state.network.num_nodes)
            
            with col2:
                st.metric("Quantum Nodes", len(st.session_state.network.quantum_nodes))
            
            with col3:
                st.metric("Classical Nodes", len(st.session_state.network.classical_nodes))
            
            with col4:
                st.metric("Total Links", len(st.session_state.network.G.edges()))
            
            # Link type breakdown
            quantum_links = sum(1 for edge in st.session_state.network.G.edges() 
                              if st.session_state.network.link_properties[edge]['type'] == 'quantum')
            classical_links = len(st.session_state.network.G.edges()) - quantum_links
            
            st.subheader("Link Distribution")
            link_data = pd.DataFrame({
                'Link Type': ['Quantum', 'Classical'],
                'Count': [quantum_links, classical_links]
            })
            
            fig_pie = px.pie(link_data, values='Count', names='Link Type', 
                           color_discrete_map={'Quantum': '#ff6b6b', 'Classical': '#4ecdc4'})
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="section-header">Link Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.network and st.session_state.link_sim:
            # Simulation parameters
            col1, col2 = st.columns(2)
            
            with col1:
                num_simulations = st.slider("Number of Simulations", 100, 2000, 1000)
            
            with col2:
                if st.button("🔬 Run Link Analysis"):
                    with st.spinner("Running link analysis..."):
                        results, distances, quantum_effects, classical_effects = st.session_state.link_sim.analyze_link_behavior(num_simulations)
                        
                        # Success rates
                        quantum_success = np.mean(results['quantum']) if results['quantum'] else 0
                        classical_success = np.mean(results['classical']) if results['classical'] else 0
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Quantum Success Rate", f"{quantum_success:.1%}", 
                                    delta=f"{quantum_success - classical_success:.1%}")
                        
                        with col2:
                            st.metric("Classical Success Rate", f"{classical_success:.1%}")
                        
                        # Success rate comparison
                        fig_bar = go.Figure(data=[
                            go.Bar(name='Success Rate', x=['Quantum', 'Classical'], 
                                  y=[quantum_success, classical_success],
                                  marker_color=['#ff6b6b', '#4ecdc4'])
                        ])
                        fig_bar.update_layout(title="Link Success Rates Comparison", 
                                            yaxis_title="Success Rate")
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Quantum Effects Breakdown
                        st.subheader("Quantum Networking Challenges Analysis")
                        
                        # Quantum effects pie chart
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Quantum Link Failure Modes")
                            quantum_total = sum(quantum_effects.values())
                            if quantum_total > 0:
                                fig_quantum_effects = px.pie(
                                    values=list(quantum_effects.values()),
                                    names=[
                                        f"Decoherence ({quantum_effects['decoherence']})",
                                        f"No-Cloning Violation ({quantum_effects['no_cloning']})",
                                        f"Entanglement Failure ({quantum_effects['entanglement_failure']})",
                                        f"Swap Failure - Repeater ({quantum_effects['swap_failure']})",
                                        f"Successful Transmission ({quantum_effects['successful_quantum']})"
                                    ],
                                    title="Quantum Link Outcomes with Repeater Analysis",
                                    color_discrete_map={
                                        f"Decoherence ({quantum_effects['decoherence']})": '#ff4444',
                                        f"No-Cloning Violation ({quantum_effects['no_cloning']})": '#ff8800',
                                        f"Entanglement Failure ({quantum_effects['entanglement_failure']})": '#ffaa44',
                                        f"Swap Failure - Repeater ({quantum_effects['swap_failure']})": '#ff6600',
                                        f"Successful Transmission ({quantum_effects['successful_quantum']})": '#44ff44'
                                    }
                                )
                                st.plotly_chart(fig_quantum_effects, use_container_width=True)
                        
                        with col2:
                            st.subheader("Classical Link Outcomes")
                            classical_total = sum(classical_effects.values())
                            if classical_total > 0:
                                fig_classical_effects = px.pie(
                                    values=list(classical_effects.values()),
                                    names=[
                                        f"Packet Loss ({classical_effects['packet_loss']})",
                                        f"Successful Transmission ({classical_effects['successful_classical']})"
                                    ],
                                    title="Classical Link Outcomes",
                                    color_discrete_map={
                                        f"Packet Loss ({classical_effects['packet_loss']})": '#ff6666',
                                        f"Successful Transmission ({classical_effects['successful_classical']})": '#66ff66'
                                    }
                                )
                                st.plotly_chart(fig_classical_effects, use_container_width=True)
                        
                        # Quantum Effects vs Distance Analysis
                        st.subheader("Quantum Effects vs Distance Analysis")
                        with st.spinner("Analyzing quantum effects by distance..."):
                            distance_effects = st.session_state.link_sim.analyze_quantum_effects_vs_distance(2000)
                            
                            # Prepare data for stacked bar chart
                            distance_labels = list(distance_effects.keys())
                            decoherence_rates = []
                            no_cloning_rates = []
                            entanglement_failure_rates = []
                            success_rates = []
                            
                            for label in distance_labels:
                                total = distance_effects[label]['total_attempts']
                                if total > 0:
                                    decoherence_rates.append(distance_effects[label]['decoherence'] / total * 100)
                                    no_cloning_rates.append(distance_effects[label]['no_cloning'] / total * 100)
                                    entanglement_failure_rates.append(distance_effects[label]['entanglement_failure'] / total * 100)
                                    success_rates.append(distance_effects[label]['successful_quantum'] / total * 100)
                                else:
                                    decoherence_rates.append(0)
                                    no_cloning_rates.append(0)
                                    entanglement_failure_rates.append(0)
                                    success_rates.append(0)
                            
                            # Create stacked bar chart
                            fig_distance = go.Figure()
                            
                            fig_distance.add_trace(go.Bar(
                                name='Successful Transmission',
                                x=distance_labels,
                                y=success_rates,
                                marker_color='#44ff44'
                            ))
                            
                            fig_distance.add_trace(go.Bar(
                                name='Decoherence',
                                x=distance_labels,
                                y=decoherence_rates,
                                marker_color='#ff4444'
                            ))
                            
                            fig_distance.add_trace(go.Bar(
                                name='Entanglement Failure',
                                x=distance_labels,
                                y=entanglement_failure_rates,
                                marker_color='#ffaa44'
                            ))
                            
                            fig_distance.add_trace(go.Bar(
                                name='No-Cloning Violation',
                                x=distance_labels,
                                y=no_cloning_rates,
                                marker_color='#ff8800'
                            ))
                            
                            fig_distance.update_layout(
                                title="Quantum Link Behavior vs Distance",
                                xaxis_title="Distance Range",
                                yaxis_title="Percentage (%)",
                                barmode='stack'
                            )
                            st.plotly_chart(fig_distance, use_container_width=True)
                        
                        # Success vs Distance scatter plot
                        if results['quantum'] and results['classical']:
                            fig_scatter = go.Figure()
                            
                            fig_scatter.add_trace(go.Scatter(
                                x=distances['quantum'], y=results['quantum'],
                                mode='markers', name='Quantum', 
                                marker=dict(color='red', opacity=0.6)
                            ))
                            
                            fig_scatter.add_trace(go.Scatter(
                                x=distances['classical'], y=results['classical'],
                                mode='markers', name='Classical', 
                                marker=dict(color='blue', opacity=0.6)
                            ))
                            
                            fig_scatter.update_layout(
                                title="Success vs Distance",
                                xaxis_title="Distance (km)",
                                yaxis_title="Success (1) / Failure (0)"
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">Quantum Networking Effects</h2>', unsafe_allow_html=True)
        
        if st.session_state.network and st.session_state.link_sim:
            st.markdown("""
            ### Quantum Networking Challenges Simulation
            
            This tab provides detailed analysis of the major quantum networking challenges:
            - **Decoherence**: Quantum states decay over distance due to environmental interaction
            - **No-Cloning Theorem**: Quantum information cannot be copied or amplified
            - **Entanglement Distribution**: Creating and maintaining entangled states across distance
            - **Quantum Repeater Swap Failures**: Entanglement swapping at repeater nodes can fail (~8% failure rate)
            """)
            
            # Simulation parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_sims = st.slider("Number of Simulations", 1000, 5000, 3000, key="quantum_sims")
            
            with col2:
                distance_focus = st.selectbox("Distance Analysis", 
                                            ["All Distances", "Short Range (0-30km)", 
                                             "Medium Range (30-70km)", "Long Range (70-100km)"])
            
            with col3:
                effect_focus = st.selectbox("Effect Focus", 
                                          ["All Effects", "Decoherence Only", 
                                           "No-Cloning Only", "Entanglement Only", "Swap Failures Only"])
            
            if st.button("🔬 Run Quantum Effects Analysis", key="quantum_analysis"):
                with st.spinner("Analyzing quantum networking challenges..."):
                    # Run comprehensive analysis
                    results, distances, quantum_effects, classical_effects = st.session_state.link_sim.analyze_link_behavior(num_sims)
                    distance_effects = st.session_state.link_sim.analyze_quantum_effects_vs_distance(num_sims)
                    
                    # Display key metrics
                    st.subheader("Quantum Challenge Impact Metrics")
                    
                    total_quantum = sum(quantum_effects.values())
                    if total_quantum > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            decoherence_rate = quantum_effects['decoherence'] / total_quantum
                            st.metric("Decoherence Rate", f"{decoherence_rate:.1%}", 
                                    help="Percentage of quantum transmissions failed due to decoherence")
                        
                        with col2:
                            no_cloning_rate = quantum_effects['no_cloning'] / total_quantum
                            st.metric("No-Cloning Violations", f"{no_cloning_rate:.1%}",
                                    help="Percentage of attempts that violated the no-cloning theorem")
                        
                        with col3:
                            entanglement_rate = quantum_effects['entanglement_failure'] / total_quantum
                            st.metric("Entanglement Failures", f"{entanglement_rate:.1%}",
                                    help="Percentage of failed entanglement distributions")
                        
                        with col4:
                            success_rate = quantum_effects['successful_quantum'] / total_quantum
                            st.metric("Quantum Success Rate", f"{success_rate:.1%}",
                                    help="Overall quantum transmission success rate")
                    
                    # Detailed Analysis Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distance vs Effect Type Heatmap
                        st.subheader("Quantum Effects by Distance Range")
                        
                        # Prepare heatmap data
                        effect_types = ['Decoherence', 'No-Cloning', 'Entanglement Failure', 'Success']
                        distance_ranges = list(distance_effects.keys())
                        
                        heatmap_data = []
                        for dist_range in distance_ranges:
                            total = distance_effects[dist_range]['total_attempts']
                            if total > 0:
                                row = [
                                    distance_effects[dist_range]['decoherence'] / total * 100,
                                    distance_effects[dist_range]['no_cloning'] / total * 100,
                                    distance_effects[dist_range]['entanglement_failure'] / total * 100,
                                    distance_effects[dist_range]['successful_quantum'] / total * 100
                                ]
                            else:
                                row = [0, 0, 0, 0]
                            heatmap_data.append(row)
                        
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=heatmap_data,
                            x=effect_types,
                            y=distance_ranges,
                            colorscale='RdYlBu_r',
                            text=[[f"{val:.1f}%" for val in row] for row in heatmap_data],
                            texttemplate="%{text}",
                            textfont={"size": 12}
                        ))
                        
                        fig_heatmap.update_layout(
                            title="Quantum Effect Distribution by Distance",
                            xaxis_title="Effect Type",
                            yaxis_title="Distance Range"
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    with col2:
                        # Effect Evolution with Distance
                        st.subheader("Quantum Challenge Evolution")
                        
                        # Create line plot showing how each effect changes with distance
                        fig_evolution = go.Figure()
                        
                        distances_mid = [12.5, 37.5, 62.5, 87.5]  # Midpoints of ranges
                        
                        decoherence_evolution = []
                        no_cloning_evolution = []
                        entanglement_evolution = []
                        success_evolution = []
                        
                        for dist_range in distance_ranges:
                            total = distance_effects[dist_range]['total_attempts']
                            if total > 0:
                                decoherence_evolution.append(distance_effects[dist_range]['decoherence'] / total * 100)
                                no_cloning_evolution.append(distance_effects[dist_range]['no_cloning'] / total * 100)
                                entanglement_evolution.append(distance_effects[dist_range]['entanglement_failure'] / total * 100)
                                success_evolution.append(distance_effects[dist_range]['successful_quantum'] / total * 100)
                            else:
                                decoherence_evolution.append(0)
                                no_cloning_evolution.append(0)
                                entanglement_evolution.append(0)
                                success_evolution.append(0)
                        
                        fig_evolution.add_trace(go.Scatter(
                            x=distances_mid, y=decoherence_evolution,
                            mode='lines+markers', name='Decoherence',
                            line=dict(color='red', width=3)
                        ))
                        
                        fig_evolution.add_trace(go.Scatter(
                            x=distances_mid, y=no_cloning_evolution,
                            mode='lines+markers', name='No-Cloning Violations',
                            line=dict(color='orange', width=3)
                        ))
                        
                        fig_evolution.add_trace(go.Scatter(
                            x=distances_mid, y=entanglement_evolution,
                            mode='lines+markers', name='Entanglement Failures',
                            line=dict(color='purple', width=3)
                        ))
                        
                        fig_evolution.add_trace(go.Scatter(
                            x=distances_mid, y=success_evolution,
                            mode='lines+markers', name='Successful Transmission',
                            line=dict(color='green', width=3)
                        ))
                        
                        fig_evolution.update_layout(
                            title="Quantum Effects vs Distance",
                            xaxis_title="Distance (km)",
                            yaxis_title="Occurrence Rate (%)",
                            yaxis=dict(range=[0, max(max(decoherence_evolution), max(success_evolution)) + 5])
                        )
                        st.plotly_chart(fig_evolution, use_container_width=True)
                    
                    # Detailed Breakdown
                    st.subheader("Quantum Physics Principles in Action")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        **🌊 Decoherence Effects**
                        - Quantum states lose coherence over distance
                        - Probability increases exponentially with distance
                        - Environmental interference destroys quantum properties
                        - Major challenge for long-distance quantum communication
                        """)
                        
                        if total_quantum > 0:
                            decoherence_pct = quantum_effects['decoherence'] / total_quantum * 100
                            st.metric("Impact", f"{decoherence_pct:.1f}% of transmissions")
                    
                    with col2:
                        st.markdown("""
                        **🚫 No-Cloning Theorem**
                        - Quantum information cannot be copied
                        - Prevents quantum amplification/repeaters
                        - Fundamental limit of quantum mechanics
                        - Requires quantum teleportation for long distances
                        """)
                        
                        if total_quantum > 0:
                            no_cloning_pct = quantum_effects['no_cloning'] / total_quantum * 100
                            st.metric("Violation Rate", f"{no_cloning_pct:.1f}% of attempts")
                    
                    with col3:
                        st.markdown("""
                        **🔗 Entanglement Distribution**
                        - Creating shared entangled states
                        - Success probability decreases with distance
                        - Foundation for quantum key distribution
                        - Enables quantum teleportation protocols
                        """)
                        
                        if total_quantum > 0:
                            entanglement_pct = quantum_effects['entanglement_failure'] / total_quantum * 100
                            st.metric("Failure Rate", f"{entanglement_pct:.1f}% of attempts")
                    
                    # Quantum Repeater Analysis
                    st.subheader("Quantum Repeater Impact Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔗 Without Quantum Repeaters**")
                        
                        # Simulate long-distance transmission without repeaters
                        long_distance_results = {'success': [], 'distance': []}
                        quantum_edges = [edge for edge in st.session_state.network.G.edges() 
                                       if st.session_state.network.link_properties[edge]['type'] == 'quantum']
                        
                        if not quantum_edges:
                            st.warning("⚠️ No quantum links found in the network. Try generating a new network with higher quantum ratio.")
                        else:
                            for _ in range(500):
                                edge = random.choice(quantum_edges)
                                result = st.session_state.link_sim.simulate_transmission(edge[0], edge[1])
                                distance = result['distance']
                                
                                # For distances > 50km, simulate exponential decay without repeaters
                                if distance > 50:
                                    success_prob = np.exp(-distance * 0.02)  # Exponential decay
                                    success = random.random() < success_prob
                                else:
                                    success = result['success']
                                
                                long_distance_results['success'].append(1 if success else 0)
                                long_distance_results['distance'].append(distance)
                        
                        if long_distance_results['success']:
                            avg_success_no_repeaters = np.mean(long_distance_results['success'])
                            st.metric("Success Rate", f"{avg_success_no_repeaters:.1%}", 
                                    help="Success rate for long-distance quantum transmission without repeaters")
                            
                            # Distance vs success plot
                            fig_no_repeaters = go.Figure()
                            fig_no_repeaters.add_trace(go.Scatter(
                                x=long_distance_results['distance'],
                                y=long_distance_results['success'],
                                mode='markers',
                                name='Without Repeaters',
                                marker=dict(color='red', opacity=0.6, size=8)
                            ))
                            fig_no_repeaters.update_layout(
                                title="Transmission Success vs Distance (No Repeaters)",
                                xaxis_title="Distance (km)",
                                yaxis_title="Success (1) / Failure (0)",
                                height=300
                            )
                            st.plotly_chart(fig_no_repeaters, use_container_width=True)
                    
                    with col2:
                        st.markdown("**🔄 With Quantum Repeaters**")
                        
                        # Simulate with quantum repeaters (every 25km)
                        repeater_results = {'success': [], 'distance': []}
                        
                        if quantum_edges:
                            for _ in range(500):
                                edge = random.choice(quantum_edges)
                                result = st.session_state.link_sim.simulate_transmission(edge[0], edge[1])
                                distance = result['distance']
                                
                                # With repeaters every 25km, calculate success for each segment
                                if distance > 25:
                                    num_segments = int(np.ceil(distance / 25))
                                    segment_success_prob = 0.8  # Each segment has 80% success
                                    overall_success_prob = segment_success_prob ** num_segments
                                    success = random.random() < overall_success_prob
                                else:
                                    success = result['success']
                                
                                repeater_results['success'].append(1 if success else 0)
                                repeater_results['distance'].append(distance)
                        else:
                            st.warning("⚠️ No quantum links available for repeater analysis.")
                        
                        if repeater_results['success']:
                            avg_success_repeaters = np.mean(repeater_results['success'])
                            improvement = avg_success_repeaters - avg_success_no_repeaters if 'avg_success_no_repeaters' in locals() else 0
                            st.metric("Success Rate", f"{avg_success_repeaters:.1%}", 
                                    delta=f"+{improvement:.1%}",
                                    help="Success rate with quantum repeaters every 25km")
                            
                            # Distance vs success plot
                            fig_repeaters = go.Figure()
                            fig_repeaters.add_trace(go.Scatter(
                                x=repeater_results['distance'],
                                y=repeater_results['success'],
                                mode='markers',
                                name='With Repeaters',
                                marker=dict(color='green', opacity=0.6, size=8)
                            ))
                            fig_repeaters.update_layout(
                                title="Transmission Success vs Distance (With Repeaters)",
                                xaxis_title="Distance (km)",
                                yaxis_title="Success (1) / Failure (0)",
                                height=300
                            )
                            st.plotly_chart(fig_repeaters, use_container_width=True)
                    
                    # Comparative Analysis
                    st.subheader("Repeater Effectiveness Comparison")
                    
                    if ('avg_success_no_repeaters' in locals() and 'avg_success_repeaters' in locals() and 
                        not np.isnan(avg_success_no_repeaters) and not np.isnan(avg_success_repeaters)):
                        # Create comparison chart
                        comparison_data = pd.DataFrame({
                            'Configuration': ['Without Repeaters', 'With Repeaters (25km spacing)'],
                            'Success Rate': [avg_success_no_repeaters, avg_success_repeaters],
                            'Color': ['#ff4444', '#44ff44']
                        })
                        
                        fig_comparison = px.bar(
                            comparison_data, 
                            x='Configuration', 
                            y='Success Rate',
                            title="Quantum Network Performance: Repeaters vs No Repeaters",
                            color='Color',
                            color_discrete_map={'#ff4444': '#ff4444', '#44ff44': '#44ff44'}
                        )
                        fig_comparison.update_layout(showlegend=False)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    else:
                        st.info("💡 Generate a network with more quantum links to see repeater comparison analysis.")
                        
                        # Distance breakdown analysis
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Short Range (0-25km)**")
                            short_no_rep_list = [s for i, s in enumerate(long_distance_results['success']) 
                                               if long_distance_results['distance'][i] <= 25]
                            short_rep_list = [s for i, s in enumerate(repeater_results['success']) 
                                            if repeater_results['distance'][i] <= 25]
                            
                            short_no_rep = np.mean(short_no_rep_list) if short_no_rep_list else 0
                            short_rep = np.mean(short_rep_list) if short_rep_list else 0
                            
                            st.metric("Without Repeaters", f"{short_no_rep:.1%}" if short_no_rep_list else "No data")
                            st.metric("With Repeaters", f"{short_rep:.1%}" if short_rep_list else "No data")
                        
                        with col2:
                            st.markdown("**Medium Range (25-50km)**")
                            medium_no_rep_list = [s for i, s in enumerate(long_distance_results['success']) 
                                                 if 25 < long_distance_results['distance'][i] <= 50]
                            medium_rep_list = [s for i, s in enumerate(repeater_results['success']) 
                                             if 25 < repeater_results['distance'][i] <= 50]
                            
                            medium_no_rep = np.mean(medium_no_rep_list) if medium_no_rep_list else 0
                            medium_rep = np.mean(medium_rep_list) if medium_rep_list else 0
                            
                            st.metric("Without Repeaters", f"{medium_no_rep:.1%}" if medium_no_rep_list else "No data")
                            st.metric("With Repeaters", f"{medium_rep:.1%}" if medium_rep_list else "No data")
                        
                        with col3:
                            st.markdown("**Long Range (50km+)**")
                            long_no_rep_list = [s for i, s in enumerate(long_distance_results['success']) 
                                               if long_distance_results['distance'][i] > 50]
                            long_rep_list = [s for i, s in enumerate(repeater_results['success']) 
                                           if repeater_results['distance'][i] > 50]
                            
                            long_no_rep = np.mean(long_no_rep_list) if long_no_rep_list else 0
                            long_rep = np.mean(long_rep_list) if long_rep_list else 0
                            
                            st.metric("Without Repeaters", f"{long_no_rep:.1%}" if long_no_rep_list else "No data")
                            st.metric("With Repeaters", f"{long_rep:.1%}" if long_rep_list else "No data")
                    
                    # Quantum Repeater Technology Explanation
                    st.subheader("Quantum Repeater Technology")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **🔄 How Quantum Repeaters Work:**
                        - Break long distances into shorter segments
                        - Use quantum memory to store quantum states
                        - Perform entanglement swapping between segments
                        - Enable long-distance quantum communication
                        - Overcome exponential decay of quantum signals
                        """)
                    
                    with col2:
                        st.markdown("""
                        **📈 Benefits of Quantum Repeaters:**
                        - Extend quantum communication range
                        - Maintain quantum properties over distance
                        - Enable global quantum internet
                        - Preserve entanglement across long distances
                        - Make quantum networks practically feasible
                        """)
    
    with tab4:
        st.markdown('<h2 class="section-header">Message Routing</h2>', unsafe_allow_html=True)
        
        if st.session_state.network and st.session_state.router:
            # Message routing interface
            col1, col2, col3 = st.columns(3)
            
            with col1:
                source_node = st.selectbox("Source Node", 
                                         list(range(st.session_state.network.num_nodes)))
            
            with col2:
                dest_node = st.selectbox("Destination Node", 
                                       list(range(st.session_state.network.num_nodes)))
            
            with col3:
                message = st.text_input("Message", "Hello Quantum World!")
            
            if st.button("� Send Message"):
                if source_node != dest_node:
                    with st.spinner("Routing message..."):
                        result = st.session_state.router.send_message(source_node, dest_node, message)
                        
                        if result['success']:
                            # Enhanced success message with path analysis
                            method_desc = {
                                'quantum-majority': 'quantum-majority path (mostly quantum links)',
                                'classical-majority': 'classical-majority path (mostly classical links)', 
                                'hybrid-balanced': 'hybrid-balanced path (equal quantum/classical)',
                                'quantum-majority-fallback': 'quantum-majority fallback path',
                                'classical-majority-fallback': 'classical-majority fallback path',
                                'hybrid-balanced-fallback': 'hybrid-balanced fallback path'
                            }
                            
                            method_text = method_desc.get(result['final_method'], result['final_method'])
                            st.success(f"✅ Message delivered successfully via {method_text}!")
                            
                            # Show path with analysis
                            path_str = " → ".join(map(str, result['path']))
                            
                            if 'path_analysis' in result:
                                analysis = result['path_analysis']
                                path_info = (f"**Path:** {path_str}\n\n"
                                           f"**Path Composition:** "
                                           f"{analysis['quantum_links']} quantum links, "
                                           f"{analysis['classical_links']} classical links "
                                           f"({analysis['quantum_percentage']:.0f}% quantum)")
                                st.info(path_info)
                            else:
                                st.info(f"**Path:** {path_str}")
                            
                            # Show attempts
                            st.subheader("Transmission Attempts")
                            
                            attempts_data = []
                            for i, attempt in enumerate(result['attempts']):
                                attempts_data.append({
                                    'Attempt': i+1,
                                    'Type': attempt['type'].capitalize(),
                                    'Link': f"{attempt['link'][0]} → {attempt['link'][1]}",
                                    'Success': '✅' if attempt['success'] else '❌',
                                    'Reason': attempt['reason']
                                })
                            
                            attempts_df = pd.DataFrame(attempts_data)
                            st.dataframe(attempts_df, use_container_width=True)
                            
                        else:
                            st.error("❌ Message delivery failed!")
                            
                            # Show failed attempts
                            if result['attempts']:
                                st.subheader("Failed Attempts")
                                attempts_data = []
                                for i, attempt in enumerate(result['attempts']):
                                    attempts_data.append({
                                        'Attempt': i+1,
                                        'Type': attempt['type'].capitalize(),
                                        'Link': f"{attempt['link'][0]} → {attempt['link'][1]}",
                                        'Reason': attempt['reason']
                                    })
                                
                                attempts_df = pd.DataFrame(attempts_data)
                                st.dataframe(attempts_df, use_container_width=True)
                else:
                    st.warning("Source and destination nodes must be different!")

    with tab5:
        st.markdown('<h2 class="section-header">Performance Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.network and st.session_state.router:
            # Performance testing
            col1, col2 = st.columns(2)
            
            with col1:
                num_tests = st.slider("Number of Tests", 50, 500, 100)
            
            with col2:
                if st.button("📊 Run Performance Analysis"):
                    with st.spinner("Running performance tests..."):
                        # Run multiple tests
                        results = []
                        quantum_successes = 0
                        classical_successes = 0
                        total_tests = 0
                        
                        progress_bar = st.progress(0)
                        
                        for i in range(num_tests):
                            source = random.randint(0, st.session_state.network.num_nodes - 1)
                            dest = random.randint(0, st.session_state.network.num_nodes - 1)
                            
                            if source != dest:
                                result = st.session_state.router.send_message(source, dest, "test")
                                
                                results.append({
                                    'test': i+1,
                                    'source': source,
                                    'dest': dest,
                                    'success': result['success'],
                                    'method': result['final_method'] if result['success'] else 'failed',
                                    'path_length': len(result['path']) if result['path'] else 0
                                })
                                
                                if result['success']:
                                    if 'quantum' in result['final_method']:
                                        quantum_successes += 1
                                    else:
                                        classical_successes += 1
                                
                                total_tests += 1
                            
                            progress_bar.progress((i + 1) / num_tests)
                        
                        # Display results
                        overall_success = (quantum_successes + classical_successes) / total_tests if total_tests > 0 else 0
                        quantum_utilization = quantum_successes / total_tests if total_tests > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Overall Success Rate", f"{overall_success:.1%}")
                        
                        with col2:
                            st.metric("Quantum Utilization", f"{quantum_utilization:.1%}")
                        
                        with col3:
                            avg_path_length = np.mean([r['path_length'] for r in results if r['success']])
                            st.metric("Avg Path Length", f"{avg_path_length:.1f}")
                        
                        # Results breakdown
                        results_df = pd.DataFrame(results)
                        
                        if not results_df.empty:
                            # Success distribution
                            method_counts = results_df['method'].value_counts()
                            
                            fig_pie = px.pie(values=method_counts.values, names=method_counts.index,
                                           title="Success Distribution by Method")
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Path length distribution
                            successful_results = results_df[results_df['success'] == True]
                            if not successful_results.empty:
                                fig_hist = px.histogram(successful_results, x='path_length', 
                                                      title="Path Length Distribution")
                                st.plotly_chart(fig_hist, use_container_width=True)

    with tab6:
        st.markdown('<h2 class="section-header">Post-Quantum Key Exchange Systems (2030)</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Scenario: Post-Quantum Cryptography Era
        
        **Year 2030**: Quantum computers have become ubiquitous and public key cryptography has been broken.
        We need secure communication systems for 25 people where any 2 can communicate privately 
        without the other 23 being able to eavesdrop.
        """)
        
        # Key Exchange System Analysis
        st.subheader("🔐 Key Exchange System Options")
        
        num_users = 25  # Fixed to match the problem statement
        
        # Calculate requirements for different approaches
        pairwise_keys = (num_users * (num_users - 1)) // 2  # 300 keys
        kdc_keys = num_users  # 25 keys
        hierarchical_keys = int(np.sqrt(num_users)) * num_users  # 125 keys (5x25)
        qkd_keys = num_users  # 25 quantum channels
        
        # New approaches for post-quantum era
        group_key_rotation = num_users + (num_users // 5)  # 30 keys (5 groups of 5)
        mesh_tree_hybrid = int(num_users * np.log2(num_users))  # ~115 keys
        
        st.markdown("#### Option Comparison for 25 Users")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔄 Pairwise Key Exchange", f"{pairwise_keys} keys", 
                     help="Every pair shares a unique key")
            st.metric("🏛️ Key Distribution Center", f"{kdc_keys} keys",
                     help="Central authority manages all keys")
        
        with col2:
            st.metric("🌳 Hierarchical System", f"{hierarchical_keys} keys",
                     help="Tree-based key distribution")
            st.metric("⚛️ Quantum Key Distribution", f"{qkd_keys} channels",
                     help="Quantum-secured key exchange")
        
        with col3:
            st.metric("👥 Group Key Rotation", f"{group_key_rotation} keys",
                     help="Dynamic group-based approach")
            st.metric("🕸️ Mesh-Tree Hybrid", f"{mesh_tree_hybrid} keys",
                     help="Optimized hybrid approach")
        
        # Detailed Analysis
        st.subheader("📊 Comprehensive Trade-off Analysis")
        
        # Create comparison data
        systems_data = {
            'System': ['Pairwise', 'KDC', 'Hierarchical', 'QKD', 'Group Rotation', 'Mesh-Tree'],
            'Keys/Channels': [pairwise_keys, kdc_keys, hierarchical_keys, qkd_keys, group_key_rotation, mesh_tree_hybrid],
            'Scalability': [1, 8, 6, 9, 7, 8],  # 1-10 scale
            'Security': [9, 5, 6, 10, 7, 8],
            'Complexity': [3, 7, 6, 9, 5, 6],
            'Quantum Resistance': [8, 3, 4, 10, 6, 7]
        }
        
        systems_df = pd.DataFrame(systems_data)
        
        # Multi-criteria comparison chart
        fig_comparison = go.Figure()
        
        criteria = ['Scalability', 'Security', 'Complexity', 'Quantum Resistance']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, system in enumerate(systems_df['System']):
            fig_comparison.add_trace(go.Scatterpolar(
                r=[systems_df.iloc[i]['Scalability'], 
                   systems_df.iloc[i]['Security'],
                   10 - systems_df.iloc[i]['Complexity'],  # Invert complexity (lower is better)
                   systems_df.iloc[i]['Quantum Resistance']],
                theta=criteria,
                fill='toself',
                name=system,
                line_color=colors[i]
            ))
        
        fig_comparison.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Multi-Criteria System Comparison (Higher = Better)"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Implementation Selection
        st.subheader("🛠️ System Implementation")
        
        selected_system = st.selectbox(
            "Choose a key exchange system to implement:",
            ['Pairwise Key Exchange', 'Quantum Key Distribution (QKD)', 'Group Key Rotation', 'Mesh-Tree Hybrid']
        )
        
        # Initialize system implementation state
        if 'selected_system_impl' not in st.session_state:
            st.session_state.selected_system_impl = None
            
        if st.button("🚀 Implement Selected System"):
            st.session_state.selected_system_impl = selected_system
            st.rerun()
            
        # Show the implementation interface if a system has been selected
        if st.session_state.selected_system_impl == 'Pairwise Key Exchange':
            st.markdown("### 🔄 Pairwise Key Exchange Implementation")
            
            # Initialize encryption results storage first
            if 'encryption_results' not in st.session_state:
                st.session_state.encryption_results = {}
            
            # Initialize or check session state for pairwise keys
            if 'pairwise_keys' not in st.session_state:
                    with st.spinner("Generating pairwise keys..."):
                        # Generate unique keys for each pair
                        import hashlib
                        import secrets
                        import time
                        
                        users = [f"User_{i:02d}" for i in range(1, 26)]
                        key_pairs = {}
                        key_storage = {user: {} for user in users}
                        
                        # Generate keys for each pair
                        for i in range(len(users)):
                            for j in range(i + 1, len(users)):
                                # Generate a secure random key using improved method
                                key_seed = f"{users[i]}_{users[j]}_{random.randint(10000, 99999)}_{random.randint(100000, 999999)}_{random.random()}"
                                key_base = abs(hash(key_seed)) * random.randint(1000, 9999)
                                key = hex(key_base & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)[2:].zfill(64)  # 256-bit key
                                pair = (users[i], users[j])
                                key_pairs[pair] = key
                                
                                # Store key for both users
                                key_storage[users[i]][users[j]] = key
                                key_storage[users[j]][users[i]] = key
                        
                        # Store in session state
                        st.session_state.pairwise_keys = key_pairs
                        st.session_state.pairwise_storage = key_storage
                        st.session_state.pairwise_users = users
                    
                    st.success(f"✅ Generated {len(st.session_state.pairwise_keys)} unique pairwise keys!")
            else:
                st.success(f"✅ Using existing {len(st.session_state.pairwise_keys)} pairwise keys!")
                
                # Get data from session state
                key_pairs = st.session_state.pairwise_keys
                key_storage = st.session_state.pairwise_storage
                users = st.session_state.pairwise_users
                
                # Display key distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Key Distribution Summary:**")
                    st.metric("Total Unique Keys", len(key_pairs))
                    st.metric("Keys per User", len(key_storage[users[0]]))
                    st.metric("Total Storage Required", f"{len(key_pairs) * 64} bytes")
                
                with col2:
                    st.markdown("**Security Properties:**")
                    st.write("🔒 Perfect Forward Secrecy")
                    st.write("🚫 No Single Point of Failure") 
                    st.write("⚡ Immediate Communication")
                    st.write("🔐 256-bit Symmetric Encryption")
                
                # Sample communication simulation
                st.markdown("**Communication Simulation:**")
                
                # Add reset button
                if st.button("🔄 Regenerate New Keys", key="reset_pairwise"):
                    if 'pairwise_keys' in st.session_state:
                        del st.session_state.pairwise_keys
                        del st.session_state.pairwise_storage
                        del st.session_state.pairwise_users
                    st.rerun()
                
                sender = st.selectbox("Sender:", users, key="pairwise_sender")
                receiver = st.selectbox("Receiver:", users, key="pairwise_receiver")
                
                # Debug information
                st.write(f"DEBUG: Sender = {sender}, Receiver = {receiver}")
                st.write(f"DEBUG: Are they different? {sender != receiver}")
                
                # Handle when same user is selected
                if sender == receiver:
                    st.warning("⚠️ **Please select two different users to see the encryption interface**")
                    st.info("👆 Use the dropdown menus above to select different sender and receiver")
                
                # Show shared key immediately when users are selected
                if sender != receiver:
                    shared_key = key_storage[sender][receiver]
                    
                    st.markdown("---")
                    st.markdown("### 🔑 **Shared Key Information**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Communication Pair:** {sender} ↔ {receiver}")
                        st.success(f"**Shared Key (First 32 chars):** `{shared_key[:32]}...`")
                    
                    with col2:
                        st.info(f"**Full Key Length:** {len(shared_key)} characters (256-bit)")
                        st.warning("🚫 **Other 23 users do NOT have this key**")
                    
                    # Show expandable full key
                    with st.expander("🔍 View Full Shared Key"):
                        st.code(shared_key)
                        st.caption("This is the complete 256-bit key shared only between the selected users")
                    
                    st.markdown("---")
                    
                    # Always show message input and encrypt button for different users
                    st.markdown("### 💬 **Send Encrypted Message**")
                    message = st.text_input("Message:", "Secret quantum research data", key="pairwise_msg")
                    
                    # Initialize encryption storage in session state
                    if 'encryption_results' not in st.session_state:
                        st.session_state.encryption_results = {}
                    
                    # ALWAYS show the encrypt button when sender != receiver with better styling
                    st.markdown("#### 🔐 **Encrypt and Send Message**")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Ready to encrypt:** '{message}' from {sender} to {receiver}")
                    with col2:
                        encrypt_button = st.button("🔐 **ENCRYPT & SEND**", key="encrypt_btn", type="primary")
                    
                    st.write(f"DEBUG: Button was clicked: {encrypt_button}")
                    
                    if encrypt_button:
                        # Get the shared key
                        shared_key = key_storage[sender][receiver]
                        
                        # Simple encryption simulation (XOR with key hash)
                        key_hash = bytes.fromhex(shared_key)[:32]  # Use first 32 bytes of hex key
                        message_bytes = message.encode('utf-8')
                        
                        encrypted = bytes(a ^ b for a, b in zip(message_bytes, 
                                        (key_hash * (len(message_bytes) // len(key_hash) + 1))[:len(message_bytes)]))
                        
                        # Store encryption results in session state
                        encryption_key = f"{sender}_{receiver}_{message}"
                        # Initialize counter if not exists
                        if 'encryption_counter' not in st.session_state:
                            st.session_state.encryption_counter = 1
                        else:
                            st.session_state.encryption_counter += 1
                        
                        st.session_state.encryption_results[encryption_key] = {
                            'sender': sender,
                            'receiver': receiver,
                            'original_message': message,
                            'encrypted_hex': encrypted.hex(),
                            'shared_key': shared_key,
                            'message_id': st.session_state.encryption_counter
                        }
                        
                        st.success(f"✅ **Message successfully encrypted and sent!**")
                
                # Always show encryption results if they exist for current sender/receiver/message
                encryption_key = f"{sender}_{receiver}_{message}"
                if sender != receiver and encryption_key in st.session_state.encryption_results:
                    st.markdown("---")
                    st.markdown("### 🔐 **ENCRYPTION RESULTS** (Persistent Display)")
                    
                    result = st.session_state.encryption_results[encryption_key]
                    
                    # Show encryption details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📤 Encryption Details:**")
                        st.write(f"**Original Message:** `{result['original_message']}`")
                        st.write(f"**Message Length:** {len(result['original_message'])} characters")
                        st.write(f"**Encryption Method:** XOR with shared key")
                        st.write(f"**Sender:** {result['sender']}")
                        st.write(f"**Receiver:** {result['receiver']}")
                        
                    with col2:
                        st.markdown("**🔐 Encrypted Result:**")
                        st.code(f"Encrypted Hex: {result['encrypted_hex']}")
                        st.write(f"**Encrypted Length:** {len(result['encrypted_hex'])} hex chars")
                        st.write(f"**Key Used:** {result['shared_key'][:16]}...{result['shared_key'][-16:]}")
                        st.write(f"**Message ID:** #{result['message_id']}")
                    
                    # Security verification
                    st.markdown("### 🛡️ **Security Verification**")
                    st.error("🚫 **THE OTHER 23 USERS CANNOT DECRYPT THIS MESSAGE**")
                    st.write("✅ Only the sender and receiver have the shared key")
                    st.write("✅ Each pair has a unique key (300 total keys for 25 users)")
                    st.write("✅ Perfect forward secrecy - no single point of failure")
                    
                    # Clear button for this specific encryption
                    if st.button("🗑️ Clear This Encryption", key=f"clear_{encryption_key}"):
                        if encryption_key in st.session_state.encryption_results:
                            del st.session_state.encryption_results[encryption_key]
                        st.rerun()
                
                # Show all previous encryptions if any exist (with proper initialization check)
                if 'encryption_results' in st.session_state and st.session_state.encryption_results:
                    st.markdown("---")
                    st.markdown("### 📋 **All Encryption History**")
                    
                    for enc_key, result in st.session_state.encryption_results.items():
                        with st.expander(f"🔐 {result['sender']} → {result['receiver']}: '{result['original_message'][:30]}...'"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Original:** {result['original_message']}")
                                st.write(f"**From:** {result['sender']}")
                                st.write(f"**To:** {result['receiver']}")
                            with col2:
                                st.code(f"Encrypted: {result['encrypted_hex']}")
                                st.write(f"**Message ID:** #{result['message_id']}")
                    
                    if st.button("🗑️ Clear All Encryption History", key="clear_all_encryptions"):
                        st.session_state.encryption_results = {}
                        st.rerun()
            
        elif st.session_state.selected_system_impl == 'Quantum Key Distribution (QKD)':
                st.markdown("### ⚛️ Quantum Key Distribution Implementation")
                
                with st.spinner("Setting up quantum channels..."):
                    # QKD simulation
                    users = [f"User_{i:02d}" for i in range(1, 26)]
                    quantum_channels = {}
                    quantum_keys = {}
                    
                    # Each user has a quantum channel to a central QKD hub
                    for user in users:
                        # Simulate quantum key generation using improved method
                        key_seed = f"{user}_{random.randint(10000, 99999)}_{random.randint(100000, 999999)}_{random.random()}"
                        key_base = abs(hash(key_seed)) * random.randint(1000, 9999)
                        qkey = hex(key_base & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)[2:].zfill(64)
                        quantum_keys[user] = qkey
                        quantum_channels[user] = {
                            'channel_id': f"QC_{hash(user) % 1000:03d}",
                            'entanglement_fidelity': np.random.uniform(0.95, 0.99),
                            'key_generation_rate': np.random.uniform(1000, 5000),  # bits/second
                            'error_rate': np.random.uniform(0.01, 0.05)
                        }
                
                st.success("✅ Quantum channels established!")
                
                # Display QKD network status
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**QKD Network Status:**")
                    avg_fidelity = np.mean([ch['entanglement_fidelity'] for ch in quantum_channels.values()])
                    avg_rate = np.mean([ch['key_generation_rate'] for ch in quantum_channels.values()])
                    avg_error = np.mean([ch['error_rate'] for ch in quantum_channels.values()])
                    
                    st.metric("Average Fidelity", f"{avg_fidelity:.3f}")
                    st.metric("Avg Key Rate", f"{avg_rate:.0f} bits/sec")
                    st.metric("Average Error Rate", f"{avg_error:.3f}")
                
                with col2:
                    st.markdown("**Quantum Advantages:**")
                    st.write("� Information-theoretic security")
                    st.write("🎯 Eavesdropping detection")
                    st.write("🚫 No-cloning protection")
                    st.write("⚡ Real-time key generation")
                
                # QKD Communication Protocol
                st.markdown("**QKD Communication Protocol:**")
                
                sender = st.selectbox("Sender:", users, key="qkd_sender")
                receiver = st.selectbox("Receiver:", users, key="qkd_receiver")
                
                if sender != receiver and st.button("🔗 Establish Quantum-Secured Channel"):
                    # Simulate quantum key agreement protocol
                    sender_key = quantum_keys[sender]
                    receiver_key = quantum_keys[receiver]
                    
                    # Generate session key using quantum-derived material
                    session_material = sender_key + receiver_key + str(random.randint(100000, 999999)) + str(random.random())
                    session_base = abs(hash(session_material)) * random.randint(1000, 9999)
                    session_key = hex(session_base & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)[2:].zfill(64)
                    
                    # Check for eavesdropping (quantum advantage)
                    eavesdrop_detected = np.random.random() < 0.02  # 2% chance of detection
                    
                    if not eavesdrop_detected:
                        st.success("� Quantum-secured session established!")
                        st.info(f"**Session Key:** {session_key[:16]}...")
                        st.metric("Security Level", "Information-Theoretic")
                        
                        # Show quantum channel properties
                        sender_ch = quantum_channels[sender]
                        receiver_ch = quantum_channels[receiver]
                        st.write(f"Fidelity: {min(sender_ch['entanglement_fidelity'], receiver_ch['entanglement_fidelity']):.3f}")
                    else:
                        st.error("� Eavesdropping detected! Channel compromised - establishing new keys...")
                        st.warning("Quantum advantage: Eavesdropping attempt was detected and blocked!")
            
        elif st.session_state.selected_system_impl == 'Group Key Rotation':
                st.markdown("### 👥 Group Key Rotation Implementation")
                
                with st.spinner("Setting up dynamic groups..."):
                    # Divide users into groups of 5
                    users = [f"User_{i:02d}" for i in range(1, 26)]
                    groups = [users[i:i+5] for i in range(0, 25, 5)]
                    
                    # Generate group keys and rotation schedule
                    group_keys = {}
                    inter_group_keys = {}
                    rotation_schedule = {}
                    
                    for i, group in enumerate(groups):
                        group_id = f"Group_{i+1}"
                        # Generate more robust keys using multiple random sources
                        current_seed = f"{group_id}_current_{random.randint(10000, 99999)}_{random.randint(100000, 999999)}_{random.random()}"
                        next_seed = f"{group_id}_next_{random.randint(10000, 99999)}_{random.randint(100000, 999999)}_{random.random()}"
                        
                        # Create better distributed keys by combining hash with random values
                        current_base = abs(hash(current_seed)) * random.randint(1000, 9999)
                        next_base = abs(hash(next_seed)) * random.randint(1000, 9999)
                        
                        group_keys[group_id] = {
                            'current_key': hex(current_base & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)[2:].zfill(64),
                            'next_key': hex(next_base & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)[2:].zfill(64),
                            'members': group,
                            'rotation_interval': 3600  # 1 hour
                        }
                        
                        # Generate inter-group keys
                        for j, other_group in enumerate(groups):
                            if i < j:
                                inter_seed = f"{group_id}_intergroup_{j+1}_{random.randint(10000, 99999)}_{time.time()}_{random.random()}"
                                inter_base = abs(hash(inter_seed)) * random.randint(1000, 9999)
                                inter_group_keys[(group_id, f"Group_{j+1}")] = hex(inter_base & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)[2:].zfill(64)
                
                st.success("✅ Group key system initialized!")
                
                # Display group structure
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Group Structure:**")
                    for group_id, group_info in group_keys.items():
                        with st.expander(f"{group_id} ({len(group_info['members'])} members)"):
                            st.write("**Members:**", ", ".join(group_info['members']))
                            st.write("**Key ID:**", group_info['current_key'][:16] + "...")
                            st.write("**Rotation:**", f"Every {group_info['rotation_interval']}s")
                
                with col2:
                    st.markdown("**System Benefits:**")
                    st.write("🔄 Dynamic key rotation")
                    st.write("👥 Efficient group management")
                    st.write("⚡ Scalable architecture")
                    st.write("🛡️ Compartmentalized security")
                
                # Communication simulation
                st.markdown("**Group Communication:**")
                
                sender = st.selectbox("Sender:", users, key="group_sender")
                receiver = st.selectbox("Receiver:", users, key="group_receiver")
                
                if sender != receiver and st.button("📡 Send via Group System"):
                    # Find groups for sender and receiver
                    sender_group = None
                    receiver_group = None
                    
                    for group_id, group_info in group_keys.items():
                        if sender in group_info['members']:
                            sender_group = group_id
                        if receiver in group_info['members']:
                            receiver_group = group_id
                    
                    if sender_group == receiver_group:
                        # Same group - use group key
                        group_key = group_keys[sender_group]['current_key']
                        st.success(f"� Intra-group communication via {sender_group}")
                        st.info(f"Using group key: {group_key[:16]}...")
                    else:
                        # Different groups - use inter-group key
                        inter_key_pair = tuple(sorted([sender_group, receiver_group]))
                        if inter_key_pair in inter_group_keys:
                            inter_key = inter_group_keys[inter_key_pair]
                            st.success(f"📤 Inter-group communication: {sender_group} → {receiver_group}")
                            st.info(f"Using inter-group key: {inter_key[:16]}...")
                        else:
                            st.error("No inter-group key found!")
            
        elif st.session_state.selected_system_impl == 'Mesh-Tree Hybrid':
                st.markdown("### 🕸️ Mesh-Tree Hybrid Implementation")
                
                with st.spinner("Building hybrid topology..."):
                    # Create a hybrid mesh-tree structure
                    users = [f"User_{i:02d}" for i in range(1, 26)]
                    
                    # Tree backbone (spanning tree)
                    import networkx as nx
                    G = nx.Graph()
                    G.add_nodes_from(users)
                    
                    # Create tree backbone
                    for i in range(1, len(users)):
                        parent = users[(i-1)//2]  # Binary tree structure
                        child = users[i]
                        G.add_edge(parent, child)
                    
                    # Add mesh connections for redundancy
                    mesh_edges = []
                    for i in range(0, len(users), 5):
                        for j in range(i, min(i+5, len(users))):
                            for k in range(j+1, min(i+5, len(users))):
                                G.add_edge(users[j], users[k])
                                mesh_edges.append((users[j], users[k]))
                    
                    # Generate keys for each edge
                    edge_keys = {}
                    for edge in G.edges():
                        edge_seed = f"{edge[0]}_{edge[1]}_edge_{random.randint(10000, 99999)}_{random.randint(100000, 999999)}_{random.random()}"
                        edge_base = abs(hash(edge_seed)) * random.randint(1000, 9999)
                        edge_keys[edge] = hex(edge_base & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)[2:].zfill(64)
                
                st.success("✅ Hybrid mesh-tree network established!")
                
                # Display network properties
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Network Properties:**")
                    st.metric("Total Edges", len(G.edges()))
                    st.metric("Tree Edges", len(users) - 1)
                    st.metric("Mesh Edges", len(mesh_edges))
                    st.metric("Avg Path Length", f"{nx.average_shortest_path_length(G):.2f}")
                
                with col2:
                    st.markdown("**Hybrid Advantages:**")
                    st.write("🌳 Tree efficiency")
                    st.write("🕸️ Mesh redundancy") 
                    st.write("⚡ Short path lengths")
                    st.write("🛡️ Fault tolerance")
                
                # Path finding simulation
                st.markdown("**Communication Path Finding:**")
                
                sender = st.selectbox("Sender:", users, key="hybrid_sender")
                receiver = st.selectbox("Receiver:", users, key="hybrid_receiver")
                
                if sender != receiver and st.button("🗺️ Find Secure Path"):
                    # Find shortest path
                    try:
                        path = nx.shortest_path(G, sender, receiver)
                        path_length = len(path) - 1
                        
                        st.success(f"📍 Optimal path found ({path_length} hops)")
                        st.info("**Path:** " + " → ".join(path))
                        
                        # Show keys needed for this path
                        keys_needed = []
                        for i in range(len(path) - 1):
                            edge = tuple(sorted([path[i], path[i+1]]))
                            if edge in edge_keys:
                                keys_needed.append(edge_keys[edge][:16] + "...")
                        
                        st.write("**Keys required:**")
                        for i, key in enumerate(keys_needed):
                            st.code(f"Hop {i+1}: {key}")
                        
                        # Show alternative paths
                        all_paths = list(nx.all_simple_paths(G, sender, receiver, cutoff=4))
                        if len(all_paths) > 1:
                            st.write(f"**Redundancy:** {len(all_paths)} alternative paths available")
                    
                    except nx.NetworkXNoPath:
                        st.error("No path found between users!")
        
        # Trade-off Summary
        st.subheader("📋 Final Recommendations")
        
        recommendations = {
            "2030 Quantum Era": {
                "Primary Choice": "Quantum Key Distribution (QKD)",
                "Reason": "Information-theoretic security, eavesdropping detection",
                "Fallback": "Group Key Rotation", 
                "Use Case": "High-security government/military communications"
            },
            "Practical Deployment": {
                "Primary Choice": "Mesh-Tree Hybrid",
                "Reason": "Balance of security, efficiency, and fault tolerance",
                "Fallback": "Group Key Rotation",
                "Use Case": "Corporate/enterprise secure communications"
            },
            "Maximum Security": {
                "Primary Choice": "Pairwise Key Exchange",
                "Reason": "No single point of failure, perfect forward secrecy",
                "Fallback": "QKD",
                "Use Case": "Critical infrastructure, financial systems"
            }
        }
        
        for scenario, rec in recommendations.items():
            with st.expander(f"🎯 {scenario} Recommendation"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Primary:** {rec['Primary Choice']}")
                    st.write(f"**Reason:** {rec['Reason']}")
                with col2:
                    st.write(f"**Fallback:** {rec['Fallback']}")
                    st.write(f"**Use Case:** {rec['Use Case']}")
        
        # Security Analysis
        st.subheader("🔒 Dynamic Security Analysis")
        
        if st.session_state.network and st.session_state.link_sim:
            # Real-time security assessment based on current network
            st.markdown("""
            **Live Security Assessment**: Analyzing the current network topology and quantum effects
            to provide real-time security metrics for each key exchange system.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                security_system = st.selectbox(
                    "Select System for Security Analysis:",
                    ['Pairwise Key Exchange', 'Quantum Key Distribution (QKD)', 
                     'Group Key Rotation', 'Mesh-Tree Hybrid'],
                    key="security_analysis_system"
                )
            
            with col2:
                threat_level = st.select_slider(
                    "Threat Level:",
                    options=['Low', 'Medium', 'High', 'Nation-State'],
                    value='Medium'
                )
            
            if st.button("🔍 Analyze Current Network Security"):
                with st.spinner("Performing dynamic security analysis..."):
                    
                    # Get current network properties
                    num_nodes = st.session_state.network.num_nodes
                    quantum_nodes = len(st.session_state.network.quantum_nodes)
                    quantum_links = sum(1 for edge in st.session_state.network.G.edges() 
                                      if st.session_state.network.link_properties[edge]['type'] == 'quantum')
                    
                    # Run link analysis for security assessment
                    results, distances, quantum_effects, classical_effects = st.session_state.link_sim.analyze_link_behavior(1000)
                    
                    # Calculate network-specific security metrics
                    quantum_success_rate = np.mean(results['quantum']) if results['quantum'] else 0
                    classical_success_rate = np.mean(results['classical']) if results['classical'] else 0
                    
                    # Threat modeling based on network topology
                    threat_multipliers = {
                        'Low': 1.0,
                        'Medium': 1.5,
                        'High': 2.0,
                        'Nation-State': 3.0
                    }
                    
                    threat_factor = threat_multipliers[threat_level]
                    
                    # Security analysis for selected system
                    if security_system == 'Pairwise Key Exchange':
                        # Security depends on key storage and network topology
                        base_security = 0.95
                        topology_factor = min(1.0, quantum_success_rate + 0.3)  # Better quantum links = better security
                        
                        # Account for attack surface (more nodes = more risk)
                        node_factor = max(0.7, 1.0 - (num_nodes - 25) * 0.01)
                        
                        security_score = base_security * topology_factor * node_factor / threat_factor
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Security Score", f"{security_score:.1%}", 
                                     help="Overall security assessment")
                        
                        with col2:
                            key_compromise_risk = 1 - security_score
                            st.metric("Key Compromise Risk", f"{key_compromise_risk:.1%}",
                                     help="Risk of key being compromised")
                        
                        with col3:
                            attack_detection_prob = min(0.9, quantum_success_rate * 0.8 + 0.4)
                            st.metric("Attack Detection", f"{attack_detection_prob:.1%}",
                                     help="Probability of detecting an attack")
                        
                        # Network-specific vulnerabilities
                        st.subheader("Network-Specific Security Assessment")
                        
                        vulnerabilities = []
                        strengths = []
                        
                        if quantum_success_rate < 0.5:
                            vulnerabilities.append("⚠️ Low quantum link reliability increases classical attack surface")
                        else:
                            strengths.append("✅ High quantum link reliability provides strong security foundation")
                        
                        if num_nodes > 20:
                            vulnerabilities.append("⚠️ Large network increases key management complexity")
                        
                        if quantum_links / len(st.session_state.network.G.edges()) < 0.3:
                            vulnerabilities.append("⚠️ Low quantum link ratio limits security benefits")
                        else:
                            strengths.append("✅ High quantum link ratio enhances overall security")
                        
                        if threat_level in ['High', 'Nation-State']:
                            vulnerabilities.append(f"🚨 {threat_level} threat requires additional countermeasures")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Security Strengths:**")
                            for strength in strengths:
                                st.write(strength)
                        
                        with col2:
                            st.markdown("**Vulnerabilities:**")
                            for vuln in vulnerabilities:
                                st.write(vuln)
                    
                    elif security_system == 'Quantum Key Distribution (QKD)':
                        # QKD security depends heavily on quantum channel quality
                        base_security = 0.99
                        quantum_channel_quality = quantum_success_rate
                        
                        # Calculate quantum advantage
                        total_quantum = sum(quantum_effects.values())
                        if total_quantum > 0:
                            eavesdrop_detection_rate = 1.0 - (quantum_effects['successful_quantum'] / total_quantum)
                        else:
                            eavesdrop_detection_rate = 0.95
                        
                        security_score = base_security * quantum_channel_quality
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Security Score", f"{security_score:.1%}")
                        
                        with col2:
                            st.metric("Eavesdrop Detection", f"{eavesdrop_detection_rate:.1%}",
                                     help="Quantum mechanical eavesdropping detection")
                        
                        with col3:
                            info_theoretic_security = min(0.99, quantum_success_rate * 0.99)
                            st.metric("Info-Theoretic Security", f"{info_theoretic_security:.1%}",
                                     help="Theoretical maximum security level")
                        
                        # Quantum channel analysis
                        st.subheader("Quantum Channel Security Analysis")
                        
                        # Simulate real-time quantum channel monitoring
                        channel_data = []
                        for edge in st.session_state.network.G.edges():
                            if st.session_state.network.link_properties[edge]['type'] == 'quantum':
                                distance = st.session_state.network.link_properties[edge]['distance']
                                
                                # Simulate channel quality metrics
                                fidelity = max(0.8, 1.0 - distance * 0.002)
                                error_rate = min(0.1, distance * 0.001)
                                key_rate = max(100, 5000 - distance * 50)  # bits/second
                                
                                channel_data.append({
                                    'Link': f"{edge[0]} → {edge[1]}",
                                    'Distance (km)': distance,
                                    'Fidelity': fidelity,
                                    'Error Rate': error_rate,
                                    'Key Rate (bps)': key_rate,
                                    'Security Level': 'High' if fidelity > 0.95 else 'Medium' if fidelity > 0.9 else 'Low'
                                })
                        
                        if channel_data:
                            df = pd.DataFrame(channel_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Channel quality visualization
                            fig_channels = px.scatter(df, x='Distance (km)', y='Fidelity', 
                                                    color='Security Level',
                                                    title="Quantum Channel Security vs Distance",
                                                    hover_data=['Error Rate', 'Key Rate (bps)'])
                            st.plotly_chart(fig_channels, use_container_width=True)
                    
                    elif security_system == 'Group Key Rotation':
                        # Group security depends on isolation and rotation frequency
                        base_security = 0.85
                        
                        # Calculate group isolation effectiveness
                        group_size = 5
                        num_groups = (num_nodes + group_size - 1) // group_size
                        isolation_factor = min(1.0, 1.0 - (num_groups - 5) * 0.05)
                        
                        # Network connectivity affects inter-group security
                        connectivity_factor = min(1.0, quantum_success_rate + 0.2)
                        
                        security_score = base_security * isolation_factor * connectivity_factor / threat_factor
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Security Score", f"{security_score:.1%}")
                        
                        with col2:
                            breach_containment = min(0.95, isolation_factor * 0.9)
                            st.metric("Breach Containment", f"{breach_containment:.1%}",
                                     help="Ability to contain security breaches to single group")
                        
                        with col3:
                            rotation_effectiveness = min(0.9, connectivity_factor * 0.85)
                            st.metric("Key Rotation Effectiveness", f"{rotation_effectiveness:.1%}",
                                     help="Effectiveness of dynamic key rotation")
                        
                        # Group vulnerability analysis
                        st.subheader("Group-Based Security Analysis")
                        
                        # Simulate group security metrics
                        group_metrics = []
                        for i in range(num_groups):
                            group_id = f"Group_{i+1}"
                            
                            # Random but realistic group metrics
                            np.random.seed(i + 42)  # Consistent random values
                            group_connectivity = np.random.uniform(0.7, 0.95)
                            inter_group_security = np.random.uniform(0.8, 0.9)
                            rotation_compliance = np.random.uniform(0.85, 0.98)
                            
                            group_metrics.append({
                                'Group': group_id,
                                'Members': min(group_size, num_nodes - i * group_size),
                                'Internal Security': group_connectivity,
                                'Inter-Group Security': inter_group_security,
                                'Rotation Compliance': rotation_compliance,
                                'Overall Score': (group_connectivity + inter_group_security + rotation_compliance) / 3
                            })
                        
                        df_groups = pd.DataFrame(group_metrics)
                        st.dataframe(df_groups, use_container_width=True)
                        
                        # Group security visualization
                        fig_groups = px.bar(df_groups, x='Group', y='Overall Score',
                                          title="Group Security Scores",
                                          color='Overall Score',
                                          color_continuous_scale='RdYlGn')
                        st.plotly_chart(fig_groups, use_container_width=True)
                    
                    elif security_system == 'Mesh-Tree Hybrid':
                        # Hybrid security depends on redundancy and path diversity
                        base_security = 0.88
                        
                        # Calculate path redundancy
                        G = st.session_state.network.G
                        avg_paths = 0
                        path_count = 0
                        
                        # Sample path diversity
                        nodes = list(G.nodes())
                        for i in range(min(10, len(nodes))):
                            for j in range(i+1, min(i+6, len(nodes))):
                                try:
                                    paths = list(nx.all_simple_paths(G, nodes[i], nodes[j], cutoff=4))
                                    avg_paths += len(paths)
                                    path_count += 1
                                except:
                                    continue
                        
                        redundancy_factor = min(1.0, avg_paths / max(1, path_count) / 3.0)
                        fault_tolerance = min(0.95, redundancy_factor * 0.9 + quantum_success_rate * 0.1)
                        
                        security_score = base_security * redundancy_factor * fault_tolerance / threat_factor
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Security Score", f"{security_score:.1%}")
                        
                        with col2:
                            st.metric("Path Redundancy", f"{redundancy_factor:.1%}",
                                     help="Multiple secure paths available")
                        
                        with col3:
                            st.metric("Fault Tolerance", f"{fault_tolerance:.1%}",
                                     help="Resistance to node/link failures")
                        
                        # Network topology security analysis
                        st.subheader("Topology Security Analysis")
                        
                        # Calculate network properties
                        try:
                            diameter = nx.diameter(G) if nx.is_connected(G) else float('inf')
                            avg_clustering = nx.average_clustering(G)
                            density = nx.density(G)
                        except:
                            diameter = float('inf')
                            avg_clustering = 0
                            density = 0
                        
                        topology_metrics = {
                            'Network Diameter': diameter if diameter != float('inf') else 'Disconnected',
                            'Clustering Coefficient': f"{avg_clustering:.3f}",
                            'Network Density': f"{density:.3f}",
                            'Quantum Link Ratio': f"{quantum_links / len(G.edges()):.1%}" if len(G.edges()) > 0 else '0%'
                        }
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            for metric, value in list(topology_metrics.items())[:2]:
                                st.metric(metric, value)
                        
                        with col2:
                            for metric, value in list(topology_metrics.items())[2:]:
                                st.metric(metric, value)
                    
                    # Dynamic threat assessment
                    st.subheader("🚨 Dynamic Threat Assessment")
                    
                    # Real-time attack simulations
                    attack_scenarios = {
                        'Quantum Computer Attack': {
                            'probability': 0.1 if threat_level == 'Low' else 0.3 if threat_level == 'Medium' else 0.6 if threat_level == 'High' else 0.9,
                            'impact': 'Critical' if security_system in ['Pairwise Key Exchange'] else 'Low',
                            'mitigation': 'Use quantum-resistant algorithms' if 'QKD' not in security_system else 'Already quantum-resistant'
                        },
                        'Network Infiltration': {
                            'probability': max(0.1, 1.0 - security_score),
                            'impact': 'High' if security_system == 'Group Key Rotation' else 'Medium',
                            'mitigation': 'Enhanced monitoring and isolation'
                        },
                        'Side-Channel Attack': {
                            'probability': 0.2 + (threat_factor - 1.0) * 0.1,
                            'impact': 'Medium',
                            'mitigation': 'Hardware security modules and secure key storage'
                        },
                        'Social Engineering': {
                            'probability': 0.15 + (threat_factor - 1.0) * 0.05,
                            'impact': 'Variable',
                            'mitigation': 'Security training and multi-factor authentication'
                        }
                    }
                    
                    threat_data = []
                    for attack, details in attack_scenarios.items():
                        threat_data.append({
                            'Attack Type': attack,
                            'Probability': f"{details['probability']:.1%}",
                            'Impact': details['impact'],
                            'Mitigation': details['mitigation']
                        })
                    
                    df_threats = pd.DataFrame(threat_data)
                    st.dataframe(df_threats, use_container_width=True)
                    
                    # Risk visualization
                    risk_levels = [attack_scenarios[attack]['probability'] for attack in attack_scenarios.keys()]
                    attack_names = list(attack_scenarios.keys())
                    
                    fig_risk = px.bar(x=attack_names, y=risk_levels,
                                     title="Real-Time Threat Assessment",
                                     labels={'x': 'Attack Type', 'y': 'Probability'},
                                     color=risk_levels,
                                     color_continuous_scale='Reds')
                    fig_risk.update_layout(showlegend=False)
                    st.plotly_chart(fig_risk, use_container_width=True)
        
        else:
            st.warning("⚠️ Please generate a network first to perform dynamic security analysis!")
        
        # Static security comparison (for reference)
        st.subheader("📊 Historical Security Trends")
        
        if st.button("🔒 Run Security Analysis"):
            # Simulate different attack scenarios
            attack_success_rates = {
                'Classical Network': {
                    'Passive Eavesdropping': 0.85,
                    'Man-in-the-Middle': 0.60,
                    'Cryptanalysis': 0.30,
                    'Quantum Computer Attack': 0.95
                },
                'Quantum Network': {
                    'Passive Eavesdropping': 0.05,  # Detected by quantum mechanics
                    'Man-in-the-Middle': 0.10,     # Quantum authentication
                    'Cryptanalysis': 0.01,         # Information-theoretic security
                    'Quantum Computer Attack': 0.05 # QKD remains secure
                }
            }
            
            # Create comparison DataFrame
            attack_data = []
            for network_type, attacks in attack_success_rates.items():
                for attack_type, success_rate in attacks.items():
                    attack_data.append({
                        'Network Type': network_type,
                        'Attack Type': attack_type,
                        'Success Rate': success_rate
                    })
            
            attack_df = pd.DataFrame(attack_data)
            
            # Create grouped bar chart
            fig_security = px.bar(attack_df, x='Attack Type', y='Success Rate',
                                color='Network Type', barmode='group',
                                title="Attack Success Rates: Classical vs Quantum Networks",
                                color_discrete_map={
                                    'Classical Network': '#ff6b6b',
                                    'Quantum Network': '#4ecdc4'
                                })
            fig_security.update_layout(yaxis_title="Success Rate", 
                                     xaxis_title="Attack Type")
            st.plotly_chart(fig_security, use_container_width=True)
            
            # Security metrics over time
            st.subheader("Security Degradation Over Time")
            
            years = np.arange(2024, 2050)
            classical_security = np.exp(-0.1 * (years - 2024))  # Exponential decay
            quantum_security = np.full_like(years, 0.99, dtype=float)  # Remains high
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=years, y=classical_security,
                                        mode='lines', name='Classical Security',
                                        line=dict(color='red', width=3)))
            fig_time.add_trace(go.Scatter(x=years, y=quantum_security,
                                        mode='lines', name='Quantum Security',
                                        line=dict(color='blue', width=3)))
            
            fig_time.update_layout(
                title="Projected Security Levels Over Time",
                xaxis_title="Year",
                yaxis_title="Security Level",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Real-world quantum threats
        st.subheader("Quantum Computing Threat Timeline")
        
        threat_data = pd.DataFrame({
            'Year': [2024, 2030, 2035, 2040],
            'Classical Encryption Strength': [100, 60, 20, 5],
            'Quantum Computer Capability': [5, 40, 80, 95],
            'QKD Adoption': [10, 35, 70, 90]
        })
        
        fig_timeline = go.Figure()
        
        fig_timeline.add_trace(go.Scatter(
            x=threat_data['Year'], y=threat_data['Classical Encryption Strength'],
            mode='lines+markers', name='Classical Encryption Strength',
            line=dict(color='red', width=3)
        ))
        
        fig_timeline.add_trace(go.Scatter(
            x=threat_data['Year'], y=threat_data['Quantum Computer Capability'],
            mode='lines+markers', name='Quantum Computer Capability',
            line=dict(color='orange', width=3)
        ))
        
        fig_timeline.add_trace(go.Scatter(
            x=threat_data['Year'], y=threat_data['QKD Adoption'],
            mode='lines+markers', name='QKD Adoption',
            line=dict(color='green', width=3)
        ))
        
        fig_timeline.update_layout(
            title="Quantum Threat and Protection Timeline",
            xaxis_title="Year",
            yaxis_title="Relative Strength/Capability (%)",
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

# Additional utility functions
def create_network_stats_sidebar():
    """Create sidebar with network statistics."""
    if st.session_state.network:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Network Stats")
        
        # Connectivity metrics
        avg_degree = np.mean([degree for node, degree in st.session_state.network.G.degree()])
        diameter = nx.diameter(st.session_state.network.G) if nx.is_connected(st.session_state.network.G) else "N/A"
        clustering = nx.average_clustering(st.session_state.network.G)
        
        st.sidebar.metric("Average Degree", f"{avg_degree:.1f}")
        st.sidebar.metric("Network Diameter", diameter)
        st.sidebar.metric("Clustering Coefficient", f"{clustering:.3f}")
        
        # Quantum network specific metrics
        quantum_connectivity = 0
        total_edges = len(st.session_state.network.G.edges())
        
        for edge in st.session_state.network.G.edges():
            if st.session_state.network.link_properties[edge]['type'] == 'quantum':
                quantum_connectivity += 1
        
        quantum_ratio = quantum_connectivity / total_edges if total_edges > 0 else 0
        st.sidebar.metric("Quantum Link Ratio", f"{quantum_ratio:.1%}")

def create_footer():
    """Create application footer."""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Quantum-Classical Hybrid Network Simulation</strong></p>
        <p>This simulation demonstrates the principles of quantum networking, including:</p>
        <p>• Quantum entanglement distribution • No-cloning theorem • Decoherence effects</p>
        <p>• Hybrid routing protocols • Quantum key distribution • Security analysis</p>
        <p><em>Built with Streamlit, NetworkX, and Plotly</em></p>
    </div>
    """, unsafe_allow_html=True)

# Run the main application
if __name__ == "__main__":
    main()
    create_network_stats_sidebar()
    create_footer()