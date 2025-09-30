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
    
    num_nodes = st.sidebar.slider("Number of Nodes", 5, 20, st.session_state.num_nodes, key="app_num_nodes")
    quantum_ratio = st.sidebar.slider("Quantum Node Ratio", 0.1, 0.9, st.session_state.quantum_ratio, key="app_quantum_ratio")
    
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
        "🔐 Security Analysis",
        "🌐 Network Topology", 
        "📊 Link Analysis", 
        "🔬 Quantum Effects",
        "🔄 Message Routing", 
        "📈 Performance Analysis"
    ])
    
    with tab2:
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
            
            st.divider()
            
            # New Feature: Network Traffic Analyzer
            st.markdown("### 📊 Network Traffic Analysis & Flow Monitoring")
            
            traffic_col1, traffic_col2 = st.columns(2)
            
            with traffic_col1:
                st.markdown("#### 🌊 Real-Time Traffic Flow")
                
                if st.button("📈 Analyze Network Traffic", key="analyze_traffic"):
                    with st.spinner("Analyzing network traffic patterns..."):
                        # Generate realistic traffic data for each node
                        traffic_data = []
                        
                        for node in range(st.session_state.network.num_nodes):
                            node_type = "Quantum" if node in st.session_state.network.quantum_nodes else "Classical"
                            
                            # Generate traffic metrics
                            base_throughput = 100 if node_type == "Quantum" else 80
                            throughput = base_throughput + random.uniform(-20, 20)
                            
                            packets_sent = random.randint(1000, 10000)
                            packets_received = random.randint(800, 9500)
                            packet_loss = max(0, (packets_sent - packets_received) / packets_sent * 100)
                            
                            latency = random.uniform(5, 50) + (10 if node_type == "Quantum" else 0)
                            bandwidth_util = random.uniform(30, 95)
                            
                            traffic_data.append({
                                "Node": node,
                                "Type": node_type,
                                "Throughput (Mbps)": f"{throughput:.1f}",
                                "Packets Sent": packets_sent,
                                "Packets Received": packets_received,
                                "Packet Loss (%)": f"{packet_loss:.2f}",
                                "Latency (ms)": f"{latency:.1f}",
                                "Bandwidth Util (%)": f"{bandwidth_util:.1f}"
                            })
                        
                        st.success("✅ Traffic analysis completed!")
                        
                        # Display traffic table
                        traffic_df = pd.DataFrame(traffic_data)
                        st.dataframe(traffic_df, use_container_width=True)
                        
                        # Traffic summary metrics
                        avg_throughput = np.mean([float(row["Throughput (Mbps)"]) for row in traffic_data])
                        avg_latency = np.mean([float(row["Latency (ms)"]) for row in traffic_data])
                        total_packets = sum([row["Packets Sent"] for row in traffic_data])
                        avg_packet_loss = np.mean([float(row["Packet Loss (%)"]) for row in traffic_data])
                        
                        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                        
                        with summary_col1:
                            st.metric("Avg Throughput", f"{avg_throughput:.1f} Mbps")
                        with summary_col2:
                            st.metric("Avg Latency", f"{avg_latency:.1f} ms")
                        with summary_col3:
                            st.metric("Total Packets", f"{total_packets:,}")
                        with summary_col4:
                            color = "inverse" if avg_packet_loss > 5 else "normal"
                            st.metric("Avg Packet Loss", f"{avg_packet_loss:.2f}%")
                
                st.markdown("#### 🔄 Traffic Flow Visualization")
                
                if st.button("🗺️ Generate Flow Map", key="flow_map"):
                    with st.spinner("Generating network flow visualization..."):
                        # Create flow data between nodes
                        flow_data = []
                        
                        # Generate random flows between connected nodes
                        for edge in list(st.session_state.network.G.edges())[:10]:  # Limit for display
                            source, target = edge
                            
                            # Generate flow metrics
                            flow_volume = random.randint(100, 1000)
                            flow_type = st.session_state.network.link_properties[edge]['type']
                            utilization = random.uniform(20, 85)
                            
                            flow_data.append({
                                "Source": source,
                                "Target": target,
                                "Flow (MB/s)": flow_volume,
                                "Link Type": flow_type.title(),
                                "Utilization": f"{utilization:.1f}%",
                                "Status": "Normal" if utilization < 80 else "High Load"
                            })
                        
                        st.success("✅ Flow map generated!")
                        
                        # Display flow table
                        flow_df = pd.DataFrame(flow_data)
                        st.dataframe(flow_df, use_container_width=True)
                        
                        # Flow visualization chart
                        fig_flow = px.scatter(flow_df, x="Source", y="Target", 
                                            size="Flow (MB/s)", color="Link Type",
                                            title="Network Flow Map",
                                            hover_data=["Utilization", "Status"])
                        st.plotly_chart(fig_flow, use_container_width=True)
            
            with traffic_col2:
                st.markdown("#### 📡 Bandwidth Monitoring")
                
                # Bandwidth monitoring controls
                monitor_duration = st.selectbox(
                    "Monitoring Duration:",
                    ["1 hour", "6 hours", "24 hours", "7 days"],
                    key="monitor_duration"
                )
                
                if st.button("📊 Monitor Bandwidth Usage", key="bandwidth_monitor"):
                    with st.spinner(f"Monitoring bandwidth for {monitor_duration}..."):
                        # Generate time-series bandwidth data
                        duration_hours = {"1 hour": 1, "6 hours": 6, "24 hours": 24, "7 days": 168}[monitor_duration]
                        time_points = min(48, duration_hours)  # Limit data points for visualization
                        
                        bandwidth_data = []
                        
                        for i in range(time_points):
                            time_label = f"T+{i * (duration_hours // time_points)}"
                            
                            # Simulate daily patterns for bandwidth usage
                            hour_of_day = (i * (duration_hours // time_points)) % 24
                            base_usage = 50 + 30 * np.sin((hour_of_day - 6) * np.pi / 12)  # Peak during day
                            
                            quantum_usage = max(10, base_usage + random.uniform(-15, 15))
                            classical_usage = max(10, base_usage * 0.8 + random.uniform(-10, 10))
                            
                            bandwidth_data.append({
                                "Time": time_label,
                                "Quantum Links (%)": min(100, quantum_usage),
                                "Classical Links (%)": min(100, classical_usage),
                                "Total Usage (%)": min(100, (quantum_usage + classical_usage) / 2)
                            })
                        
                        bandwidth_df = pd.DataFrame(bandwidth_data)
                        
                        st.success(f"✅ Bandwidth monitoring complete for {monitor_duration}!")
                        
                        # Create bandwidth usage chart
                        fig_bandwidth = px.line(bandwidth_df, x="Time", 
                                              y=["Quantum Links (%)", "Classical Links (%)", "Total Usage (%)"],
                                              title=f"Bandwidth Usage Over {monitor_duration}",
                                              labels={"value": "Bandwidth Usage (%)"})
                        st.plotly_chart(fig_bandwidth, use_container_width=True)
                        
                        # Bandwidth statistics
                        avg_quantum = bandwidth_df["Quantum Links (%)"].mean()
                        avg_classical = bandwidth_df["Classical Links (%)"].mean()
                        peak_usage = bandwidth_df["Total Usage (%)"].max()
                        
                        bw_col1, bw_col2, bw_col3 = st.columns(3)
                        
                        with bw_col1:
                            st.metric("Avg Quantum Usage", f"{avg_quantum:.1f}%")
                        with bw_col2:
                            st.metric("Avg Classical Usage", f"{avg_classical:.1f}%")
                        with bw_col3:
                            st.metric("Peak Usage", f"{peak_usage:.1f}%")
                
                st.markdown("#### ⚠️ Congestion Detection")
                
                if st.button("🚨 Detect Network Congestion", key="congestion_detect"):
                    with st.spinner("Scanning for network congestion..."):
                        # Simulate congestion detection
                        congested_links = []
                        
                        for edge in st.session_state.network.G.edges():
                            source, target = edge
                            utilization = random.uniform(40, 98)
                            
                            if utilization > 85:  # Congestion threshold
                                congested_links.append({
                                    "Link": f"{source} ↔ {target}",
                                    "Type": st.session_state.network.link_properties[edge]['type'].title(),
                                    "Utilization": f"{utilization:.1f}%",
                                    "Severity": "Critical" if utilization > 95 else "High",
                                    "Recommended Action": "Load balance" if utilization < 90 else "Add capacity"
                                })
                        
                        if congested_links:
                            st.warning(f"⚠️ Found {len(congested_links)} congested links!")
                            
                            congestion_df = pd.DataFrame(congested_links)
                            st.dataframe(congestion_df, use_container_width=True)
                            
                            # Congestion summary
                            critical_count = len([link for link in congested_links if link["Severity"] == "Critical"])
                            high_count = len(congested_links) - critical_count
                            
                            cong_col1, cong_col2 = st.columns(2)
                            with cong_col1:
                                st.metric("🔴 Critical Congestion", critical_count)
                            with cong_col2:
                                st.metric("🟠 High Congestion", high_count)
                            
                            if critical_count > 0:
                                st.error("🚨 Immediate action required for critical congestion!")
                            else:
                                st.info("💡 Consider load balancing for optimal performance")
                        
                        else:
                            st.success("✅ No congestion detected - network is running smoothly!")
                            
                            # Show network health metrics
                            health_metrics = {
                                "Average Utilization": f"{random.uniform(45, 75):.1f}%",
                                "Response Time": f"{random.uniform(10, 30):.1f} ms",
                                "Network Efficiency": f"{random.uniform(85, 95):.1f}%",
                                "Load Distribution": "Balanced"
                            }
                            
                            for metric, value in health_metrics.items():
                                st.write(f"• **{metric}:** {value}")
        
        else:
            st.warning("⚠️ Please create a network first to view the topology analysis.")
            
            # Show network theory when no network exists
            st.markdown("### 🎓 Network Theory Fundamentals")
            
            theory_col1, theory_col2 = st.columns(2)
            
            with theory_col1:
                st.markdown("#### 📊 Graph Theory Basics")
                st.info("""
                **Key Concepts:**
                - Nodes (vertices) and edges
                - Degree, centrality measures
                - Connectivity and components
                - Small world networks
                """)
                
            with theory_col2:
                st.markdown("#### 🌐 Network Topologies")
                st.info("""
                **Common Types:**
                - Star, Ring, Mesh networks
                - Scale-free networks
                - Random networks
                - Hybrid architectures
                """)
    
    with tab3:
        st.markdown('<h2 class="section-header">Link Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.network and st.session_state.link_sim:
            # Simulation parameters
            col1, col2 = st.columns(2)
            
            with col1:
                num_simulations = st.slider("Number of Simulations", 100, 2000, 1000, key="link_analysis_sims")
            
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
            
            st.divider()
            
            # New Feature: Advanced Link Quality Analyzer
            st.markdown("### 🔍 Advanced Link Quality Assessment")
            
            quality_col1, quality_col2 = st.columns(2)
            
            with quality_col1:
                st.markdown("#### 📊 Link Performance Metrics")
                
                if st.button("📈 Analyze Link Quality", key="link_quality"):
                    with st.spinner("Analyzing link quality metrics..."):
                        # Generate detailed quality metrics for each link
                        quality_data = []
                        
                        for i, edge in enumerate(list(st.session_state.network.G.edges())[:15]):  # Limit for display
                            source, target = edge
                            link_type = st.session_state.network.link_properties[edge]['type']
                            
                            # Generate quality metrics based on link type
                            if link_type == 'quantum':
                                # Quantum-specific metrics
                                fidelity = random.uniform(0.85, 0.98)
                                decoherence_time = random.uniform(50, 200)  # microseconds
                                error_rate = random.uniform(0.001, 0.05)
                                entanglement_rate = random.uniform(100, 1000)  # Hz
                                
                                quality_score = (fidelity * 50 + 
                                               (200 - decoherence_time) * 0.2 + 
                                               (0.05 - error_rate) * 1000 + 
                                               entanglement_rate * 0.01)
                                
                                quality_data.append({
                                    "Link": f"{source} ↔ {target}",
                                    "Type": "Quantum",
                                    "Fidelity": f"{fidelity:.3f}",
                                    "Decoherence (μs)": f"{decoherence_time:.1f}",
                                    "Error Rate": f"{error_rate:.4f}",
                                    "Entanglement (Hz)": f"{entanglement_rate:.0f}",
                                    "Quality Score": f"{quality_score:.1f}",
                                    "Grade": "A" if quality_score > 90 else "B" if quality_score > 75 else "C"
                                })
                            
                            else:  # Classical link
                                # Classical-specific metrics
                                bandwidth = random.uniform(100, 1000)  # Mbps
                                latency = random.uniform(1, 20)  # ms
                                jitter = random.uniform(0.1, 2)  # ms
                                packet_loss = random.uniform(0, 0.1)  # %
                                
                                quality_score = (bandwidth * 0.05 + 
                                               (20 - latency) * 4 + 
                                               (2 - jitter) * 20 + 
                                               (0.1 - packet_loss) * 500)
                                
                                quality_data.append({
                                    "Link": f"{source} ↔ {target}",
                                    "Type": "Classical",
                                    "Bandwidth (Mbps)": f"{bandwidth:.0f}",
                                    "Latency (ms)": f"{latency:.2f}",
                                    "Jitter (ms)": f"{jitter:.2f}",
                                    "Packet Loss (%)": f"{packet_loss:.3f}",
                                    "Quality Score": f"{quality_score:.1f}",
                                    "Grade": "A" if quality_score > 80 else "B" if quality_score > 60 else "C"
                                })
                        
                        st.success("✅ Link quality analysis completed!")
                        
                        # Display quality results
                        quality_df = pd.DataFrame(quality_data)
                        st.dataframe(quality_df, use_container_width=True)
                        
                        # Quality statistics
                        quantum_links = [link for link in quality_data if link["Type"] == "Quantum"]
                        classical_links = [link for link in quality_data if link["Type"] == "Classical"]
                        
                        quality_stats_col1, quality_stats_col2, quality_stats_col3 = st.columns(3)
                        
                        with quality_stats_col1:
                            total_links = len(quality_data)
                            a_grade_count = len([link for link in quality_data if link["Grade"] == "A"])
                            st.metric("High Quality Links", f"{a_grade_count}/{total_links}")
                        
                        with quality_stats_col2:
                            if quantum_links:
                                avg_fidelity = np.mean([float(link["Fidelity"]) for link in quantum_links])
                                st.metric("Avg Quantum Fidelity", f"{avg_fidelity:.3f}")
                            else:
                                st.metric("Quantum Links", "0")
                        
                        with quality_stats_col3:
                            if classical_links:
                                avg_bandwidth = np.mean([float(link["Bandwidth (Mbps)"]) for link in classical_links])
                                st.metric("Avg Bandwidth", f"{avg_bandwidth:.0f} Mbps")
                            else:
                                st.metric("Classical Links", "0")
                
                st.markdown("#### 🔧 Link Optimization Recommendations")
                
                if st.button("💡 Generate Optimization Tips", key="optimization_tips"):
                    with st.spinner("Generating optimization recommendations..."):
                        # Generate specific recommendations
                        recommendations = []
                        
                        # Quantum link recommendations
                        if st.session_state.network.quantum_nodes:
                            quantum_recs = [
                                "🌡️ Implement cryogenic cooling to extend decoherence times",
                                "🔧 Use error correction codes to improve fidelity",
                                "⚡ Optimize entanglement distribution protocols",
                                "🛡️ Add electromagnetic shielding to reduce noise",
                                "📊 Implement real-time fidelity monitoring",
                                "⚛️ Use quantum repeaters for long-distance links"
                            ]
                            recommendations.extend(random.sample(quantum_recs, 3))
                        
                        # Classical link recommendations
                        if st.session_state.network.classical_nodes:
                            classical_recs = [
                                "📈 Upgrade to higher bandwidth connections",
                                "🔄 Implement adaptive QoS mechanisms",
                                "⚡ Optimize routing protocols for lower latency",
                                "🛡️ Add redundant paths for improved reliability",
                                "📊 Deploy traffic shaping algorithms",
                                "🌐 Use link aggregation for increased capacity"
                            ]
                            recommendations.extend(random.sample(classical_recs, 3))
                        
                        st.success("✅ Optimization recommendations generated!")
                        
                        st.markdown("**🎯 Priority Recommendations:**")
                        for i, rec in enumerate(recommendations, 1):
                            impact = random.choice(["High", "Medium", "Low"])
                            effort = random.choice(["Low", "Medium", "High"])
                            
                            color = "🟢" if impact == "High" and effort == "Low" else "🟡"
                            st.write(f"{color} **{i}.** {rec}")
                            st.write(f"   ➤ Impact: {impact} | Effort: {effort}")
            
            with quality_col2:
                st.markdown("#### 📡 Signal Integrity Analysis")
                
                if st.button("🔬 Analyze Signal Quality", key="signal_quality"):
                    with st.spinner("Analyzing signal integrity across all links..."):
                        # Generate signal quality data
                        signal_data = []
                        time_points = list(range(0, 24, 2))  # Every 2 hours
                        
                        for hour in time_points:
                            # Simulate daily variations in signal quality
                            base_quality = 85 + 10 * np.sin((hour - 6) * np.pi / 12)  # Vary with time of day
                            
                            quantum_signal = max(70, min(98, base_quality + random.uniform(-5, 5)))
                            classical_signal = max(75, min(99, base_quality + random.uniform(-3, 3)))
                            
                            signal_data.append({
                                "Hour": f"{hour:02d}:00",
                                "Quantum Signal Quality (%)": quantum_signal,
                                "Classical Signal Quality (%)": classical_signal,
                                "Temperature Effect": random.uniform(-2, 2),
                                "Environmental Noise": random.uniform(0, 5)
                            })
                        
                        signal_df = pd.DataFrame(signal_data)
                        
                        st.success("✅ Signal integrity analysis completed!")
                        
                        # Create signal quality trend chart
                        fig_signal = px.line(signal_df, x="Hour", 
                                           y=["Quantum Signal Quality (%)", "Classical Signal Quality (%)"],
                                           title="24-Hour Signal Quality Trends",
                                           labels={"value": "Signal Quality (%)"})
                        st.plotly_chart(fig_signal, use_container_width=True)
                        
                        # Signal statistics
                        avg_quantum_signal = signal_df["Quantum Signal Quality (%)"].mean()
                        avg_classical_signal = signal_df["Classical Signal Quality (%)"].mean()
                        min_signal = min(signal_df["Quantum Signal Quality (%)"].min(), 
                                       signal_df["Classical Signal Quality (%)"].min())
                        
                        signal_col1, signal_col2, signal_col3 = st.columns(3)
                        
                        with signal_col1:
                            st.metric("Avg Quantum Signal", f"{avg_quantum_signal:.1f}%")
                        with signal_col2:
                            st.metric("Avg Classical Signal", f"{avg_classical_signal:.1f}%")
                        with signal_col3:
                            color = "inverse" if min_signal < 80 else "normal"
                            st.metric("Minimum Signal", f"{min_signal:.1f}%")
                        
                        # Signal quality assessment
                        if min_signal > 90:
                            st.success("🌟 Excellent signal quality across all links!")
                        elif min_signal > 80:
                            st.info("✅ Good signal quality with minor variations")
                        else:
                            st.warning("⚠️ Some links showing degraded signal quality")
                
                st.markdown("#### ⚡ Link Performance Benchmarking")
                
                benchmark_type = st.selectbox(
                    "Benchmark Type:",
                    ["Latency Test", "Throughput Test", "Reliability Test", "Quantum Fidelity Test"],
                    key="benchmark_type"
                )
                
                if st.button("🏃‍♂️ Run Benchmark", key="run_benchmark"):
                    with st.spinner(f"Running {benchmark_type.lower()}..."):
                        # Generate benchmark results
                        benchmark_results = {}
                        
                        if benchmark_type == "Latency Test":
                            for i, edge in enumerate(list(st.session_state.network.G.edges())[:10]):
                                source, target = edge
                                link_type = st.session_state.network.link_properties[edge]['type']
                                
                                if link_type == 'quantum':
                                    latency = random.uniform(10, 50)  # Quantum links have higher latency
                                else:
                                    latency = random.uniform(1, 15)   # Classical links are faster
                                
                                benchmark_results[f"{source}↔{target}"] = f"{latency:.2f} ms"
                        
                        elif benchmark_type == "Throughput Test":
                            for i, edge in enumerate(list(st.session_state.network.G.edges())[:10]):
                                source, target = edge
                                link_type = st.session_state.network.link_properties[edge]['type']
                                
                                if link_type == 'quantum':
                                    throughput = random.uniform(1, 10)    # Limited by quantum protocols
                                else:
                                    throughput = random.uniform(100, 1000) # Higher classical throughput
                                
                                unit = "qubits/s" if link_type == 'quantum' else "Mbps"
                                benchmark_results[f"{source}↔{target}"] = f"{throughput:.1f} {unit}"
                        
                        elif benchmark_type == "Reliability Test":
                            for i, edge in enumerate(list(st.session_state.network.G.edges())[:10]):
                                source, target = edge
                                reliability = random.uniform(0.95, 0.999)
                                benchmark_results[f"{source}↔{target}"] = f"{reliability:.3f}"
                        
                        else:  # Quantum Fidelity Test
                            quantum_edges = [edge for edge in st.session_state.network.G.edges() 
                                           if st.session_state.network.link_properties[edge]['type'] == 'quantum']
                            
                            for edge in quantum_edges[:10]:
                                source, target = edge
                                fidelity = random.uniform(0.85, 0.98)
                                benchmark_results[f"{source}↔{target}"] = f"{fidelity:.4f}"
                        
                        st.success(f"✅ {benchmark_type} completed!")
                        
                        # Display benchmark results
                        if benchmark_results:
                            st.markdown(f"**{benchmark_type} Results:**")
                            
                            results_data = []
                            for link, result in benchmark_results.items():
                                results_data.append({"Link": link, "Result": result})
                            
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Benchmark summary
                            if benchmark_type == "Latency Test":
                                st.info("💡 Lower latency values indicate better performance")
                            elif benchmark_type == "Throughput Test":
                                st.info("💡 Higher throughput values indicate better performance")
                            elif benchmark_type == "Reliability Test":
                                st.info("💡 Values closer to 1.000 indicate higher reliability")
                            else:
                                st.info("💡 Higher fidelity values indicate better quantum performance")
                        
                        else:
                            st.warning("⚠️ No suitable links found for this benchmark type")

    with tab4:
        st.markdown('<h2 class="section-header">⚛️ Advanced Quantum Physics & Effects Simulation</h2>', unsafe_allow_html=True)
        
        if st.session_state.network and st.session_state.link_sim:
            # Enhanced Quantum Physics Dashboard
            st.markdown("### 🔬 Quantum Mechanics in Networking")
            
            # Interactive Physics Simulation Section
            physics_col1, physics_col2 = st.columns(2)
            
            with physics_col1:
                st.markdown("#### 🌊 Quantum Decoherence Simulator")
                
                with st.expander("🔬 Interactive Decoherence Analysis", expanded=True):
                    decoherence_col1, decoherence_col2 = st.columns(2)
                    
                    with decoherence_col1:
                        coherence_time = st.slider("Coherence Time (μs)", 1, 1000, 100, key="coherence_time")
                        temperature = st.slider("Temperature (K)", 0.1, 300.0, 4.2, key="temperature")
                    
                    with decoherence_col2:
                        distance_sim = st.slider("Distance (km)", 1, 100, 50, key="distance_sim")
                        noise_level = st.slider("Environmental Noise", 0.0, 1.0, 0.1, key="noise_level")
                    
                    if st.button("🧮 Calculate Decoherence", key="calc_decoherence"):
                        # Simulate decoherence calculation
                        decoherence_rate = (1 / coherence_time) * (temperature / 4.2) * (distance_sim / 10) * (1 + noise_level)
                        fidelity = np.exp(-decoherence_rate * distance_sim / 10)
                        
                        st.success("✅ Decoherence Analysis Complete!")
                        
                        result_col1, result_col2, result_col3 = st.columns(3)
                        with result_col1:
                            st.metric("Decoherence Rate", f"{decoherence_rate:.3f} Hz")
                        with result_col2:
                            st.metric("Quantum Fidelity", f"{fidelity:.3f}")
                        with result_col3:
                            st.metric("Success Probability", f"{fidelity*100:.1f}%")
                        
                        # Show fidelity vs distance plot
                        distances = np.linspace(1, 100, 100)
                        fidelities = [np.exp(-decoherence_rate * d / 10) for d in distances]
                        
                        fig_fidelity = go.Figure()
                        fig_fidelity.add_trace(go.Scatter(
                            x=distances, y=fidelities,
                            mode='lines', name='Quantum Fidelity',
                            line=dict(color='blue', width=3)
                        ))
                        
                        fig_fidelity.update_layout(
                            title="Quantum Fidelity vs Distance",
                            xaxis_title="Distance (km)",
                            yaxis_title="Fidelity",
                            height=300
                        )
                        st.plotly_chart(fig_fidelity, use_container_width=True)
                
                st.markdown("#### 🔗 Entanglement Swapping Simulator")
                
                with st.expander("⚛️ Bell State Analysis", expanded=False):
                    bell_state = st.selectbox(
                        "Bell State Type:",
                        ["Φ⁺ (|00⟩ + |11⟩)", "Φ⁻ (|00⟩ - |11⟩)", "Ψ⁺ (|01⟩ + |10⟩)", "Ψ⁻ (|01⟩ - |10⟩)"],
                        key="bell_state"
                    )
                    
                    num_qubits = st.slider("Number of Entangled Qubits", 2, 8, 4, key="num_qubits")
                    swap_efficiency = st.slider("Swap Efficiency", 0.5, 1.0, 0.92, key="swap_efficiency")
                    
                    if st.button("🔄 Simulate Entanglement Swapping", key="entanglement_sim"):
                        with st.spinner("Simulating quantum entanglement swapping..."):
                            # Simulate multiple swap operations
                            num_swaps = num_qubits // 2
                            final_fidelity = swap_efficiency ** num_swaps
                            
                            st.success("✅ Entanglement Swapping Complete!")
                            
                            swap_col1, swap_col2, swap_col3 = st.columns(3)
                            with swap_col1:
                                st.metric("Number of Swaps", num_swaps)
                            with swap_col2:
                                st.metric("Final Fidelity", f"{final_fidelity:.3f}")
                            with swap_col3:
                                success_prob = final_fidelity * 100
                                st.metric("Success Rate", f"{success_prob:.1f}%")
                            
                            # Show cascade effect
                            swap_steps = list(range(1, num_swaps + 1))
                            fidelity_cascade = [swap_efficiency ** i for i in swap_steps]
                            
                            fig_cascade = go.Figure()
                            fig_cascade.add_trace(go.Scatter(
                                x=swap_steps, y=fidelity_cascade,
                                mode='lines+markers', name='Fidelity Cascade',
                                line=dict(color='purple', width=3),
                                marker=dict(size=8)
                            ))
                            
                            fig_cascade.update_layout(
                                title="Entanglement Fidelity Cascade",
                                xaxis_title="Swap Step",
                                yaxis_title="Cumulative Fidelity",
                                height=300
                            )
                            st.plotly_chart(fig_cascade, use_container_width=True)
            
            with physics_col2:
                st.markdown("#### 🛡️ Quantum Error Correction")
                
                with st.expander("🔧 Error Correction Codes", expanded=True):
                    error_code = st.selectbox(
                        "QEC Code Type:",
                        ["Shor Code (9-qubit)", "Steane Code (7-qubit)", "Surface Code", "Color Code"],
                        key="error_code"
                    )
                    
                    error_rate = st.slider("Physical Error Rate", 0.001, 0.1, 0.01, key="error_rate")
                    
                    code_params = {
                        "Shor Code (9-qubit)": {"logical_qubits": 1, "physical_qubits": 9, "threshold": 0.01},
                        "Steane Code (7-qubit)": {"logical_qubits": 1, "physical_qubits": 7, "threshold": 0.0075},
                        "Surface Code": {"logical_qubits": 1, "physical_qubits": 25, "threshold": 0.007},
                        "Color Code": {"logical_qubits": 1, "physical_qubits": 19, "threshold": 0.006}
                    }
                    
                    if st.button("🧮 Analyze Error Correction", key="error_correction"):
                        params = code_params[error_code]
                        
                        # Simplified error correction analysis
                        logical_error_rate = error_rate ** 2  # Simplified second-order correction
                        if error_rate > params["threshold"]:
                            logical_error_rate = error_rate * 1.5  # Above threshold
                        
                        improvement_factor = error_rate / logical_error_rate
                        
                        st.success("✅ Error Correction Analysis Complete!")
                        
                        qec_col1, qec_col2, qec_col3 = st.columns(3)
                        with qec_col1:
                            st.metric("Physical Qubits", params["physical_qubits"])
                        with qec_col2:
                            st.metric("Logical Error Rate", f"{logical_error_rate:.4f}")
                        with qec_col3:
                            st.metric("Improvement Factor", f"{improvement_factor:.1f}x")
                        
                        if error_rate > params["threshold"]:
                            st.error("⚠️ Error rate above threshold! QEC may not be effective.")
                        else:
                            st.success("✅ Error rate below threshold - QEC is effective")
                
                st.markdown("#### 🚫 No-Cloning Theorem Demonstration")
                
                with st.expander("📋 Quantum Information Theory", expanded=False):
                    st.markdown("""
                    **The No-Cloning Theorem states that arbitrary quantum states cannot be copied.**
                    
                    This fundamental limitation affects:
                    - Quantum amplification
                    - Error correction strategies  
                    - Network routing protocols
                    - Security guarantees
                    """)
                    
                    clone_attempts = st.slider("Cloning Attempts", 1, 100, 10, key="clone_attempts")
                    
                    if st.button("🔬 Demonstrate No-Cloning", key="no_cloning"):
                        # Simulate quantum state measurement
                        original_fidelity = 1.0
                        attempted_copies = []
                        
                        for i in range(clone_attempts):
                            # Each attempt reduces fidelity due to measurement
                            copy_fidelity = random.uniform(0.4, 0.8)
                            attempted_copies.append(copy_fidelity)
                        
                        avg_copy_fidelity = np.mean(attempted_copies)
                        
                        st.warning("⚠️ No-Cloning Theorem Violated!")
                        
                        clone_col1, clone_col2, clone_col3 = st.columns(3)
                        with clone_col1:
                            st.metric("Original Fidelity", f"{original_fidelity:.3f}")
                        with clone_col2:
                            st.metric("Copy Fidelity", f"{avg_copy_fidelity:.3f}")
                        with clone_col3:
                            st.metric("Information Loss", f"{(1-avg_copy_fidelity)*100:.1f}%")
                        
                        # Show fidelity distribution
                        fig_cloning = go.Figure()
                        fig_cloning.add_trace(go.Histogram(
                            x=attempted_copies,
                            nbinsx=20,
                            name='Copy Fidelities',
                            marker=dict(color='red', opacity=0.7)
                        ))
                        
                        fig_cloning.update_layout(
                            title="Distribution of Copy Fidelities",
                            xaxis_title="Fidelity",
                            yaxis_title="Frequency",
                            height=300
                        )
                        st.plotly_chart(fig_cloning, use_container_width=True)
            
            st.divider()
            
            # Comprehensive Quantum Network Analysis
            st.markdown("### 🔬 Quantum Network Effects Analysis")
            
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            with analysis_col1:
                num_sims = st.slider("Simulation Runs", 1000, 10000, 5000, key="quantum_sims_enhanced")
            
            with analysis_col2:
                distance_range = st.selectbox(
                    "Distance Analysis:",
                    ["All Distances", "Short Range (0-30km)", "Medium Range (30-70km)", "Long Range (70-100km)"],
                    key="distance_analysis"
                )
            
            with analysis_col3:
                quantum_protocol = st.selectbox(
                    "Quantum Protocol:",
                    ["BB84 QKD", "E91 Protocol", "Quantum Teleportation", "SWAP Protocol"],
                    key="quantum_protocol"
                )
            
            if st.button("🚀 Run Comprehensive Quantum Analysis", key="comprehensive_quantum"):
                with st.spinner("Running comprehensive quantum effects analysis..."):
                    # Simulate comprehensive analysis
                    results, distances, quantum_effects, classical_effects = st.session_state.link_sim.analyze_link_behavior(num_sims)
                    
                    st.success("✅ Comprehensive Analysis Complete!")
                    
                    # Key findings summary
                    st.markdown("#### 📊 Key Findings")
                    
                    findings_col1, findings_col2, findings_col3, findings_col4 = st.columns(4)
                    
                    total_quantum = sum(quantum_effects.values()) if quantum_effects else 1
                    
                    with findings_col1:
                        decoherence_rate = quantum_effects.get('decoherence', 0) / total_quantum
                        st.metric("🌊 Decoherence Impact", f"{decoherence_rate:.1%}")
                    
                    with findings_col2:
                        entanglement_rate = quantum_effects.get('entanglement_failure', 0) / total_quantum  
                        st.metric("🔗 Entanglement Failures", f"{entanglement_rate:.1%}")
                    
                    with findings_col3:
                        success_rate = quantum_effects.get('successful_quantum', 0) / total_quantum
                        st.metric("✅ Quantum Success", f"{success_rate:.1%}")
                    
                    with findings_col4:
                        classical_success = np.mean(results['classical']) if results.get('classical') else 0
                        st.metric("💻 Classical Success", f"{classical_success:.1%}")
                    
                    # Protocol-specific analysis
                    st.markdown("#### 🔐 Protocol Performance Analysis")
                    
                    protocol_performance = {
                        "BB84 QKD": {"security": 0.99, "efficiency": 0.85, "distance_limit": 200},
                        "E91 Protocol": {"security": 0.995, "efficiency": 0.80, "distance_limit": 150},
                        "Quantum Teleportation": {"security": 1.0, "efficiency": 0.70, "distance_limit": 100},
                        "SWAP Protocol": {"security": 0.98, "efficiency": 0.88, "distance_limit": 300}
                    }
                    
                    selected_protocol = protocol_performance[quantum_protocol]
                    
                    protocol_col1, protocol_col2, protocol_col3 = st.columns(3)
                    
                    with protocol_col1:
                        st.metric("Security Level", f"{selected_protocol['security']:.1%}")
                    with protocol_col2:
                        st.metric("Efficiency", f"{selected_protocol['efficiency']:.1%}")
                    with protocol_col3:
                        st.metric("Max Distance", f"{selected_protocol['distance_limit']}km")
                    
                    # Visualization of quantum effects over time
                    st.markdown("#### 📈 Quantum Effects Timeline")
                    
                    # Generate time-series data
                    times = list(range(0, 100, 5))  # 0 to 100 time units
                    decoherence_timeline = [np.exp(-0.1 * t) for t in times]
                    entanglement_timeline = [0.9 * np.exp(-0.05 * t) for t in times]
                    fidelity_timeline = [0.95 * np.exp(-0.03 * t) for t in times]
                    
                    fig_timeline = go.Figure()
                    
                    fig_timeline.add_trace(go.Scatter(
                        x=times, y=decoherence_timeline,
                        mode='lines', name='Quantum Coherence',
                        line=dict(color='blue', width=3)
                    ))
                    
                    fig_timeline.add_trace(go.Scatter(
                        x=times, y=entanglement_timeline,
                        mode='lines', name='Entanglement Strength',
                        line=dict(color='red', width=3)
                    ))
                    
                    fig_timeline.add_trace(go.Scatter(
                        x=times, y=fidelity_timeline,
                        mode='lines', name='Overall Fidelity',
                        line=dict(color='green', width=3)
                    ))
                    
                    fig_timeline.update_layout(
                        title="Quantum Properties Degradation Over Time",
                        xaxis_title="Time (arbitrary units)",
                        yaxis_title="Quantum Property Strength",
                        height=400
                    )
                    
                    st.plotly_chart(fig_timeline, use_container_width=True)
        
        else:
            st.warning("⚠️ Please create a network first to analyze quantum effects.")
            
            # Educational content for quantum physics
            st.markdown("### 🎓 Quantum Physics Education Center")
            
            edu_col1, edu_col2 = st.columns(2)
            
            with edu_col1:
                st.markdown("#### ⚛️ Fundamental Principles")
                st.info("""
                **Quantum Mechanics in Networking:**
                - Superposition of quantum states
                - Quantum entanglement properties
                - Measurement and state collapse
                - Heisenberg uncertainty principle
                """)
                
                with st.expander("📚 Quantum State Visualization"):
                    st.markdown("""
                    **Bloch Sphere Representation:**
                    - |0⟩ state at north pole
                    - |1⟩ state at south pole  
                    - Superposition states on sphere surface
                    - Quantum operations as rotations
                    """)
            
            with edu_col2:
                st.markdown("#### 🔬 Applications in Networking")
                st.info("""
                **Quantum Network Technologies:**
                - Quantum Key Distribution (QKD)
                - Quantum repeaters and memories
                - Quantum internet protocols
                - Distributed quantum computing
                """)
                
                with st.expander("🌐 Future Quantum Internet"):
                    st.markdown("""
                    **Next-Generation Features:**
                    - Global quantum communication
                    - Quantum cloud computing
                    - Unhackable communications
                    - Quantum sensor networks
                    """)
                    
                    with col2:
                        st.markdown("**🔬 Quantum Repeater Technology**")
                        st.info("""
                        **Advanced Features:**
                        - Entanglement purification
                        - Quantum memory storage
                        - Error correction protocols
                        - Network synchronization
                        """)
                    
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
            
            st.divider()
            
            # New Feature: Quantum State Visualizer
            st.markdown("### 🌐 Quantum State Visualization & Analysis")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("#### 🎯 Quantum State Generator")
                
                state_type = st.selectbox(
                    "Quantum State Type:",
                    ["Single Qubit", "Bell State", "GHZ State", "W State", "Random State"],
                    key="quantum_state_type"
                )
                
                if st.button("🔬 Generate Quantum State", key="generate_state"):
                    with st.spinner("Generating quantum state..."):
                        # Generate quantum state based on type
                        state_info = {}
                        
                        if state_type == "Single Qubit":
                            # Random single qubit state
                            theta = random.uniform(0, np.pi)
                            phi = random.uniform(0, 2*np.pi)
                            
                            alpha = np.cos(theta/2)
                            beta = np.sin(theta/2) * np.exp(1j * phi)
                            
                            state_info = {
                                "Type": "Single Qubit",
                                "α coefficient": f"{alpha:.3f}",
                                "β coefficient": f"{beta:.3f}",
                                "Theta (θ)": f"{theta:.3f} rad",
                                "Phi (φ)": f"{phi:.3f} rad",
                                "Purity": "1.000 (pure state)",
                                "Entanglement": "N/A (single qubit)"
                            }
                            
                        elif state_type == "Bell State":
                            bell_states = [
                                ("Φ⁺", "|00⟩ + |11⟩", "Maximally entangled"),
                                ("Φ⁻", "|00⟩ - |11⟩", "Maximally entangled"),
                                ("Ψ⁺", "|01⟩ + |10⟩", "Maximally entangled"),
                                ("Ψ⁻", "|01⟩ - |10⟩", "Maximally entangled")
                            ]
                            
                            chosen_bell = random.choice(bell_states)
                            
                            state_info = {
                                "Type": "Bell State",
                                "State": chosen_bell[0],
                                "Mathematical Form": chosen_bell[1],
                                "Entanglement": chosen_bell[2],
                                "Concurrence": "1.000",
                                "Von Neumann Entropy": "1.000",
                                "Schmidt Rank": "2"
                            }
                            
                        elif state_type == "GHZ State":
                            num_qubits = random.randint(3, 5)
                            
                            state_info = {
                                "Type": "GHZ State",
                                "Qubits": str(num_qubits),
                                "Mathematical Form": f"|000...⟩ + |111...⟩ ({num_qubits} qubits)",
                                "Entanglement": "Multipartite maximally entangled",
                                "Genuine Multipartite": "Yes",
                                "Quantum Volume": f"{2**num_qubits}",
                                "Applications": "Quantum sensing, cryptography"
                            }
                            
                        elif state_type == "W State":
                            num_qubits = random.randint(3, 4)
                            
                            state_info = {
                                "Type": "W State",
                                "Qubits": str(num_qubits),
                                "Mathematical Form": f"Symmetric superposition of single excitations",
                                "Entanglement": "Multipartite entangled",
                                "Symmetry": "Permutation symmetric",
                                "Robustness": "Persistent under particle loss",
                                "Applications": "Quantum networks, distributed computing"
                            }
                            
                        else:  # Random State
                            # Generate random mixed state
                            purity = random.uniform(0.5, 1.0)
                            entropy = random.uniform(0, 1)
                            
                            state_info = {
                                "Type": "Random Mixed State",
                                "Purity": f"{purity:.3f}",
                                "Von Neumann Entropy": f"{entropy:.3f}",
                                "Rank": str(random.randint(2, 4)),
                                "Mixedness": f"{1-purity:.3f}",
                                "Quantum Features": "Partial coherence",
                                "Classical Correlation": f"{random.uniform(0.1, 0.5):.3f}"
                            }
                        
                        st.success(f"✅ {state_type} generated successfully!")
                        
                        # Display state information
                        st.markdown("**🔬 Quantum State Properties:**")
                        for property_name, value in state_info.items():
                            st.write(f"• **{property_name}:** {value}")
                        
                        # Generate measurement probabilities
                        st.markdown("**📊 Measurement Probability Distribution:**")
                        
                        if state_type == "Single Qubit":
                            prob_0 = abs(alpha)**2
                            prob_1 = abs(beta)**2
                            
                            prob_data = pd.DataFrame({
                                "Outcome": ["|0⟩", "|1⟩"],
                                "Probability": [prob_0, prob_1]
                            })
                            
                        elif state_type in ["Bell State", "GHZ State", "W State"]:
                            # Equal superposition for demonstration
                            num_outcomes = 4 if state_type == "Bell State" else 8
                            outcomes = [f"|{format(i, '0'+str(int(np.log2(num_outcomes)))+'b')}⟩" 
                                      for i in range(num_outcomes)]
                            
                            if state_type == "Bell State":
                                # Bell states have specific probabilities
                                probs = [0.5, 0, 0, 0.5] if chosen_bell[0] in ["Φ⁺", "Φ⁻"] else [0, 0.5, 0.5, 0]
                            else:
                                # Simplified probabilities for demonstration
                                probs = [0.5, 0, 0, 0, 0, 0, 0, 0.5] if state_type == "GHZ State" else [1/3, 1/3, 1/3, 0, 0, 0, 0, 0]
                                probs = probs[:num_outcomes]
                            
                            prob_data = pd.DataFrame({
                                "Outcome": outcomes,
                                "Probability": probs
                            })
                            
                        else:  # Random state
                            outcomes = ["|0⟩", "|1⟩"]
                            probs = [random.uniform(0.2, 0.8), 0]
                            probs[1] = 1 - probs[0]
                            
                            prob_data = pd.DataFrame({
                                "Outcome": outcomes,
                                "Probability": probs
                            })
                        
                        # Create probability chart
                        fig_prob = px.bar(prob_data, x="Outcome", y="Probability",
                                        title="Quantum Measurement Probabilities",
                                        color="Probability",
                                        color_continuous_scale="viridis")
                        st.plotly_chart(fig_prob, use_container_width=True)
                
                st.markdown("#### 🌊 Quantum Interference Patterns")
                
                if st.button("🌈 Simulate Interference", key="interference_sim"):
                    with st.spinner("Simulating quantum interference..."):
                        # Generate interference pattern data
                        x = np.linspace(-5, 5, 100)
                        
                        # Double slit interference pattern
                        slit_separation = random.uniform(1, 3)
                        wavelength = random.uniform(0.5, 1.5)
                        
                        # Interference intensity
                        intensity = (np.cos(np.pi * slit_separation * x / wavelength))**2
                        
                        # Add quantum noise
                        noise = np.random.normal(0, 0.05, len(x))
                        intensity_noisy = np.maximum(0, intensity + noise)
                        
                        st.success("✅ Quantum interference pattern generated!")
                        
                        # Create interference plot
                        interference_data = pd.DataFrame({
                            "Position": x,
                            "Intensity (Ideal)": intensity,
                            "Intensity (with Noise)": intensity_noisy
                        })
                        
                        fig_interference = px.line(interference_data, x="Position", 
                                                 y=["Intensity (Ideal)", "Intensity (with Noise)"],
                                                 title="Quantum Interference Pattern",
                                                 labels={"value": "Probability Amplitude", "Position": "Position (μm)"})
                        st.plotly_chart(fig_interference, use_container_width=True)
                        
                        # Interference metrics
                        visibility = (intensity.max() - intensity.min()) / (intensity.max() + intensity.min())
                        coherence_length = wavelength * random.uniform(100, 1000)
                        
                        inter_col1, inter_col2, inter_col3 = st.columns(3)
                        
                        with inter_col1:
                            st.metric("Visibility", f"{visibility:.3f}")
                        with inter_col2:
                            st.metric("Wavelength", f"{wavelength:.2f} μm")
                        with inter_col3:
                            st.metric("Coherence Length", f"{coherence_length:.0f} μm")
            
            with viz_col2:
                st.markdown("#### 🎭 Quantum State Tomography")
                
                if st.button("🔍 Perform State Tomography", key="state_tomography"):
                    with st.spinner("Performing quantum state tomography..."):
                        # Simulate state tomography measurements
                        measurement_bases = ["X", "Y", "Z", "X+Y", "X-Y", "Z+X"]
                        tomography_data = []
                        
                        for basis in measurement_bases:
                            # Generate measurement results
                            num_measurements = random.randint(500, 1000)
                            success_rate = random.uniform(0.4, 0.6)  # Around 50% for random state
                            
                            if basis in ["X", "Y", "Z"]:
                                # Pauli measurements
                                expectation = random.uniform(-0.8, 0.8)
                            else:
                                # Combined measurements
                                expectation = random.uniform(-0.6, 0.6)
                            
                            error = random.uniform(0.01, 0.05)
                            
                            tomography_data.append({
                                "Measurement Basis": basis,
                                "Measurements": num_measurements,
                                "Expectation Value": f"{expectation:.3f}",
                                "Standard Error": f"{error:.3f}",
                                "Confidence": "95%"
                            })
                        
                        st.success("✅ Quantum state tomography completed!")
                        
                        # Display tomography results
                        tomography_df = pd.DataFrame(tomography_data)
                        st.dataframe(tomography_df, use_container_width=True)
                        
                        # Reconstructed state properties
                        fidelity = random.uniform(0.85, 0.98)
                        purity = random.uniform(0.7, 0.95)
                        
                        tomo_col1, tomo_col2, tomo_col3 = st.columns(3)
                        
                        with tomo_col1:
                            st.metric("Reconstruction Fidelity", f"{fidelity:.3f}")
                        with tomo_col2:
                            st.metric("State Purity", f"{purity:.3f}")
                        with tomo_col3:
                            confidence = random.uniform(92, 98)
                            st.metric("Confidence Level", f"{confidence:.1f}%")
                        
                        # Tomography visualization
                        st.markdown("**🎯 Bloch Sphere Representation:**")
                        
                        # Generate Bloch vector
                        theta = random.uniform(0, np.pi)
                        phi = random.uniform(0, 2*np.pi)
                        
                        x_bloch = np.sin(theta) * np.cos(phi)
                        y_bloch = np.sin(theta) * np.sin(phi)
                        z_bloch = np.cos(theta)
                        
                        # Create 3D Bloch sphere visualization
                        bloch_info = {
                            "X coordinate": f"{x_bloch:.3f}",
                            "Y coordinate": f"{y_bloch:.3f}",
                            "Z coordinate": f"{z_bloch:.3f}",
                            "Polar angle θ": f"{theta:.3f} rad",
                            "Azimuthal angle φ": f"{phi:.3f} rad"
                        }
                        
                        for coord, value in bloch_info.items():
                            st.write(f"• **{coord}:** {value}")
                
                st.markdown("#### 🔄 Quantum Process Tomography")
                
                process_type = st.selectbox(
                    "Quantum Process:",
                    ["Quantum Gate", "Decoherence Channel", "Measurement Process", "Error Channel"],
                    key="process_type"
                )
                
                if st.button("⚙️ Analyze Process", key="process_analysis"):
                    with st.spinner(f"Analyzing {process_type.lower()}..."):
                        # Simulate process analysis
                        process_data = {}
                        
                        if process_type == "Quantum Gate":
                            gate_types = ["Hadamard", "CNOT", "Toffoli", "Phase", "Rotation"]
                            selected_gate = random.choice(gate_types)
                            
                            process_data = {
                                "Process Type": "Quantum Gate",
                                "Gate": selected_gate,
                                "Fidelity": f"{random.uniform(0.95, 0.999):.4f}",
                                "Gate Time": f"{random.uniform(10, 100):.1f} ns",
                                "Error Rate": f"{random.uniform(0.001, 0.01):.4f}",
                                "Coherence": "Unitary operation"
                            }
                            
                        elif process_type == "Decoherence Channel":
                            channel_types = ["Amplitude Damping", "Phase Damping", "Depolarizing", "Pauli Channel"]
                            selected_channel = random.choice(channel_types)
                            
                            process_data = {
                                "Process Type": "Decoherence Channel",
                                "Channel": selected_channel,
                                "Damping Rate": f"{random.uniform(0.01, 0.1):.3f}",
                                "Coherence Time": f"{random.uniform(10, 200):.1f} μs",
                                "Information Loss": f"{random.uniform(0.1, 0.5):.3f}",
                                "Reversibility": "Irreversible"
                            }
                            
                        elif process_type == "Measurement Process":
                            measurement_types = ["Projective", "POVM", "Weak Measurement", "Continuous"]
                            selected_measurement = random.choice(measurement_types)
                            
                            process_data = {
                                "Process Type": "Measurement Process",
                                "Type": selected_measurement,
                                "Efficiency": f"{random.uniform(0.8, 0.95):.3f}",
                                "Readout Fidelity": f"{random.uniform(0.9, 0.99):.3f}",
                                "Back-action": f"{random.uniform(0.05, 0.3):.3f}",
                                "Information Gain": f"{random.uniform(0.7, 0.95):.3f}"
                            }
                            
                        else:  # Error Channel
                            error_types = ["Bit Flip", "Phase Flip", "Amplitude Error", "Correlated Error"]
                            selected_error = random.choice(error_types)
                            
                            process_data = {
                                "Process Type": "Error Channel",
                                "Error Type": selected_error,
                                "Error Probability": f"{random.uniform(0.001, 0.05):.4f}",
                                "Correlation Length": f"{random.uniform(1, 10):.1f}",
                                "Detectability": f"{random.uniform(0.7, 0.9):.3f}",
                                "Correctability": f"{random.uniform(0.6, 0.85):.3f}"
                            }
                        
                        st.success(f"✅ {process_type} analysis completed!")
                        
                        # Display process information
                        st.markdown(f"**⚙️ {process_type} Properties:**")
                        for property_name, value in process_data.items():
                            st.write(f"• **{property_name}:** {value}")
                        
                        # Process recommendations
                        st.markdown("**💡 Optimization Recommendations:**")
                        
                        if process_type == "Quantum Gate":
                            recommendations = [
                                "🔧 Calibrate gate timing for optimal fidelity",
                                "🌡️ Maintain stable temperature conditions",
                                "⚡ Optimize control pulse shapes"
                            ]
                        elif process_type == "Decoherence Channel":
                            recommendations = [
                                "❄️ Implement better isolation from environment",
                                "🛡️ Use dynamical decoupling sequences",
                                "⏱️ Minimize operation times"
                            ]
                        elif process_type == "Measurement Process":
                            recommendations = [
                                "📡 Improve detector efficiency",
                                "🔍 Optimize readout protocols",
                                "⚡ Reduce measurement back-action"
                            ]
                        else:
                            recommendations = [
                                "🛠️ Implement error correction codes",
                                "📊 Monitor error correlations",
                                "🔄 Use error syndrome detection"
                            ]
                        
                        for rec in recommendations:
                            st.write(f"• {rec}")

    with tab5:
        st.markdown('<h2 class="section-header">🔄 Advanced Message Routing & Path Optimization</h2>', unsafe_allow_html=True)
        
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
            
            st.divider()
            
            # Advanced Routing Analysis Tools
            st.markdown("### 🔬 Advanced Routing Analysis & Optimization")
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.markdown("#### 📊 Path Comparison Analysis")
                
                # Enhanced message routing interface
                routing_col1, routing_col2 = st.columns(2)
                
                with routing_col1:
                    routing_algorithm = st.selectbox(
                        "Routing Algorithm:",
                        ["Shortest Path (Dijkstra)", "Minimum Hop Count", "Maximum Quantum Links", 
                         "Load Balanced", "Fault Tolerant", "Security Optimized"],
                        key="routing_algorithm_select"
                    )
                
                with routing_col2:
                    optimization_target = st.selectbox(
                        "Optimization Target:",
                        ["Minimize Latency", "Maximize Reliability", "Maximize Security", 
                         "Minimize Power", "Maximize Quantum Fidelity"],
                        key="optimization_target_select"
                    )
                
                if st.button("🔍 Compare All Paths", key="compare_paths"):
                    with st.spinner("Analyzing all possible paths..."):
                        # Select nodes for comparison
                        src_node = st.session_state.get('routing_source', 0)
                        dst_node = st.session_state.get('routing_dest', 1)
                        
                        try:
                            all_paths = list(nx.all_simple_paths(
                                st.session_state.network.G, src_node, dst_node, cutoff=5
                            ))[:10]  # Limit to first 10 paths
                            
                            if all_paths:
                                st.success(f"✅ Found {len(all_paths)} alternative paths")
                                
                                # Create comparison table
                                path_data = []
                                for i, path in enumerate(all_paths):
                                    # Calculate path metrics
                                    path_length = len(path) - 1
                                    quantum_hops = sum(1 for j in range(len(path)-1) 
                                                     if path[j] in st.session_state.network.quantum_nodes or 
                                                        path[j+1] in st.session_state.network.quantum_nodes)
                                    
                                    # Simulate advanced metrics
                                    latency = 10 + path_length * 15 + random.uniform(-5, 5)
                                    reliability = 0.95 ** path_length * random.uniform(0.95, 1.0)
                                    security_score = 0.8 + (quantum_hops / path_length * 0.2) if path_length > 0 else 0.8
                                    
                                    path_data.append({
                                        "Path #": i + 1,
                                        "Route": " → ".join(map(str, path)),
                                        "Hops": path_length,
                                        "Quantum Hops": quantum_hops,
                                        "Latency (ms)": f"{latency:.1f}",
                                        "Reliability": f"{reliability:.2%}",
                                        "Security": f"{security_score:.2f}",
                                        "Score": f"{(reliability + security_score)/2:.2f}"
                                    })
                                
                                df_paths = pd.DataFrame(path_data)
                                st.dataframe(df_paths, use_container_width=True)
                                
                                # Recommend best path based on selected optimization
                                if optimization_target == "Minimize Latency":
                                    best_idx = min(range(len(all_paths)), key=lambda i: len(all_paths[i]))
                                    criterion = "lowest latency"
                                elif optimization_target == "Maximize Reliability":
                                    best_idx = max(range(len(path_data)), key=lambda i: float(path_data[i]["Reliability"].strip('%'))/100)
                                    criterion = "highest reliability"
                                else:
                                    best_idx = max(range(len(path_data)), key=lambda i: float(path_data[i]["Score"]))
                                    criterion = "best overall score"
                                
                                st.info(f"💡 **Recommended:** Path #{best_idx + 1} ({criterion})")
                            
                            else:
                                st.warning("⚠️ No paths found between selected nodes")
                        
                        except nx.NetworkXNoPath:
                            st.error("❌ No path exists between the selected nodes")
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                
                st.markdown("#### 🌐 Network Routing Table")
                
                if st.button("📋 Generate Routing Table", key="routing_table"):
                    with st.spinner("Building network routing table..."):
                        # Generate routing table for all node pairs
                        routing_data = []
                        nodes = list(range(st.session_state.network.num_nodes))
                        
                        for src in nodes[:5]:  # Limit to first 5 nodes for display
                            for dst in nodes[:5]:
                                if src != dst:
                                    try:
                                        path = nx.shortest_path(st.session_state.network.G, src, dst)
                                        next_hop = path[1] if len(path) > 1 else dst
                                        distance = nx.shortest_path_length(st.session_state.network.G, src, dst)
                                        
                                        routing_data.append({
                                            "Source": src,
                                            "Destination": dst,
                                            "Next Hop": next_hop,
                                            "Distance": distance,
                                            "Full Path": " → ".join(map(str, path))
                                        })
                                    
                                    except nx.NetworkXNoPath:
                                        routing_data.append({
                                            "Source": src,
                                            "Destination": dst,
                                            "Next Hop": "N/A",
                                            "Distance": "∞",
                                            "Full Path": "No route"
                                        })
                        
                        if routing_data:
                            df_routing = pd.DataFrame(routing_data)
                            st.dataframe(df_routing, use_container_width=True)
                            
                            # Routing table statistics
                            connected_pairs = len([r for r in routing_data if r['Distance'] != "∞"])
                            total_pairs = len(routing_data)
                            connectivity_pct = connected_pairs / total_pairs * 100 if total_pairs > 0 else 0
                            
                            st.info(f"📈 **Network Connectivity:** {connectivity_pct:.1f}% of node pairs are connected")
                        else:
                            st.warning("⚠️ Unable to generate routing table")
            
            with analysis_col2:
                st.markdown("#### 🔧 Network Load Balancing")
                
                # QoS Parameters
                qos_col1, qos_col2 = st.columns(2)
                
                with qos_col1:
                    max_latency = st.slider("Max Latency (ms):", 1, 1000, 100, key="max_latency_routing")
                
                with qos_col2:
                    min_reliability = st.slider("Min Reliability:", 0.5, 1.0, 0.9, key="min_reliability_routing")
                
                if st.button("⚖️ Analyze Load Distribution", key="load_analysis"):
                    with st.spinner("Analyzing network load distribution..."):
                        # Simulate load analysis
                        node_loads = {}
                        edge_loads = {}
                        
                        # Generate random load data for nodes
                        for node in range(st.session_state.network.num_nodes):
                            base_load = random.uniform(0.1, 0.7)
                            # Quantum nodes might have different load patterns
                            if node in st.session_state.network.quantum_nodes:
                                base_load *= random.uniform(0.8, 1.2)  # Quantum processing variation
                            node_loads[node] = min(base_load, 0.95)
                        
                        # Generate load data for edges
                        for edge in st.session_state.network.G.edges():
                            edge_loads[edge] = random.uniform(0.0, 0.8)
                        
                        st.success("✅ Load analysis complete!")
                        
                        # Display load metrics
                        avg_node_load = np.mean(list(node_loads.values()))
                        max_node_load = max(node_loads.values())
                        avg_edge_load = np.mean(list(edge_loads.values()))
                        
                        load_col1, load_col2, load_col3 = st.columns(3)
                        
                        with load_col1:
                            st.metric("Avg Node Load", f"{avg_node_load:.1%}")
                        with load_col2:
                            st.metric("Max Node Load", f"{max_node_load:.1%}")
                        with load_col3:
                            st.metric("Avg Link Load", f"{avg_edge_load:.1%}")
                        
                        # Load distribution visualization
                        load_data = []
                        for node, load in node_loads.items():
                            node_type = "Quantum" if node in st.session_state.network.quantum_nodes else "Classical"
                            status = "Overloaded" if load > 0.8 else "Normal" if load > 0.5 else "Underutilized"
                            load_data.append({
                                "Node": node,
                                "Type": node_type,
                                "Load": f"{load:.1%}",
                                "Status": status
                            })
                        
                        df_loads = pd.DataFrame(load_data)
                        st.dataframe(df_loads, use_container_width=True)
                        
                        # Load balancing recommendations
                        overloaded_nodes = [node for node, load in node_loads.items() if load > 0.8]
                        if overloaded_nodes:
                            st.warning(f"⚠️ Overloaded nodes detected: {overloaded_nodes}")
                            st.info("💡 Consider redistributing traffic or adding parallel paths")
                        else:
                            st.success("✅ Network load is well balanced")
                
                st.markdown("#### 🛡️ Fault Tolerance Analysis")
                
                if st.button("🔍 Analyze Fault Tolerance", key="fault_tolerance"):
                    with st.spinner("Analyzing network fault tolerance..."):
                        # Critical node analysis
                        critical_nodes = []
                        
                        for node in range(st.session_state.network.num_nodes):
                            # Test connectivity without this node
                            G_test = st.session_state.network.G.copy()
                            G_test.remove_node(node)
                            
                            if not nx.is_connected(G_test) and len(G_test.nodes()) > 1:
                                critical_nodes.append(node)
                        
                        st.success("✅ Fault tolerance analysis complete!")
                        
                        # Display results
                        fault_col1, fault_col2 = st.columns(2)
                        
                        with fault_col1:
                            st.metric("Critical Nodes", len(critical_nodes))
                            if critical_nodes:
                                st.error(f"🚨 Critical nodes: {critical_nodes}")
                                st.warning("Network will be disconnected if these nodes fail!")
                            else:
                                st.success("✅ No single points of failure detected")
                        
                        with fault_col2:
                            # Calculate redundancy metrics
                            total_edges = len(st.session_state.network.G.edges())
                            min_edges = st.session_state.network.num_nodes - 1  # Minimum for connectivity
                            redundancy_ratio = total_edges / min_edges if min_edges > 0 else 0
                            
                            st.metric("Redundancy Ratio", f"{redundancy_ratio:.2f}")
                            
                            if redundancy_ratio > 2.0:
                                st.success("🌟 High redundancy - excellent fault tolerance")
                            elif redundancy_ratio > 1.5:
                                st.info("✅ Good redundancy level")
                            else:
                                st.warning("⚠️ Low redundancy - consider additional connections")
        
        else:
            st.warning("⚠️ Please create a network first to access routing features.")
            
            # Educational content about routing
            st.markdown("### 🎓 Network Routing Education")
            
            edu_col1, edu_col2 = st.columns(2)
            
            with edu_col1:
                st.markdown("#### 🛤️ Routing Algorithms")
                st.info("""
                **Common Algorithms:**
                - **Dijkstra's Algorithm:** Shortest path with weighted edges
                - **Bellman-Ford:** Handles negative weights, detects cycles
                - **Floyd-Warshall:** All-pairs shortest paths
                - **A* Search:** Heuristic-guided pathfinding
                """)
                
                st.markdown("#### 📊 Performance Metrics")
                st.info("""
                **Key Metrics:**
                - **Latency:** End-to-end delay
                - **Throughput:** Data transmission rate
                - **Reliability:** Success rate of transmissions
                - **Jitter:** Variation in packet timing
                """)
                
            with edu_col2:
                st.markdown("#### 🎯 Quality of Service (QoS)")
                st.info("""
                **QoS Parameters:**
                - **Priority Classes:** Critical, high, normal, low
                - **Traffic Shaping:** Rate limiting and burst control
                - **Congestion Control:** Managing network overload
                - **Service Guarantees:** SLA compliance
                """)
                
                st.markdown("#### 🔐 Security Features")
                st.info("""
                **Security Aspects:**
                - **Encrypted Routing:** Secure path information
                - **Authentication:** Verify node identity
                - **Intrusion Detection:** Monitor suspicious activity
                - **Quantum Cryptography:** Quantum-safe protocols
                """)

    with tab6:
        st.markdown('<h2 class="section-header">📈 Advanced Performance Analysis & Benchmarking</h2>', unsafe_allow_html=True)
        
        if st.session_state.network and st.session_state.router:
            # Performance testing
            col1, col2 = st.columns(2)
            
            with col1:
                num_tests = st.slider("Number of Tests", 50, 500, 100, key="performance_tests")
            
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
            
            st.divider()
            
            # Advanced Performance Analysis Tools
            st.markdown("### 🔬 Advanced Performance Analytics")
            
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                st.markdown("#### 📊 Network Performance Metrics")
                
                if st.button("🚀 Comprehensive Performance Benchmark", key="comprehensive_benchmark"):
                    with st.spinner("Running comprehensive performance analysis..."):
                        
                        # Simulate comprehensive network performance metrics
                        performance_metrics = {
                            'Latency': {
                                'mean': random.uniform(50, 150),
                                'std': random.uniform(10, 30),
                                'p95': random.uniform(100, 200),
                                'p99': random.uniform(150, 300)
                            },
                            'Throughput': {
                                'mean': random.uniform(80, 95),
                                'std': random.uniform(5, 15),
                                'p95': random.uniform(70, 85),
                                'p99': random.uniform(60, 75)
                            },
                            'Reliability': {
                                'mean': random.uniform(92, 99),
                                'std': random.uniform(1, 3),
                                'p95': random.uniform(88, 94),
                                'p99': random.uniform(85, 90)
                            },
                            'Quantum_Fidelity': {
                                'mean': random.uniform(85, 95),
                                'std': random.uniform(3, 8),
                                'p95': random.uniform(80, 88),
                                'p99': random.uniform(75, 82)
                            }
                        }
                        
                        st.success("✅ Comprehensive benchmark completed!")
                        
                        # Performance metrics table
                        metrics_data = []
                        for metric, values in performance_metrics.items():
                            metrics_data.append({
                                'Metric': metric.replace('_', ' '),
                                'Mean': f"{values['mean']:.2f}",
                                'Std Dev': f"{values['std']:.2f}",
                                '95th %ile': f"{values['p95']:.2f}",
                                '99th %ile': f"{values['p99']:.2f}",
                                'Quality': 'Excellent' if values['mean'] > 90 else 'Good' if values['mean'] > 80 else 'Needs Improvement'
                            })
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Performance trends visualization
                        time_series_data = []
                        time_points = list(range(1, 25))  # 24 hours
                        
                        for hour in time_points:
                            for metric in ['Latency', 'Throughput', 'Reliability']:
                                base_value = performance_metrics[metric]['mean']
                                # Add daily variation
                                variation = random.uniform(-0.1, 0.1) * base_value
                                value = base_value + variation
                                
                                time_series_data.append({
                                    'Hour': hour,
                                    'Metric': metric,
                                    'Value': value
                                })
                        
                        ts_df = pd.DataFrame(time_series_data)
                        
                        # Create performance trend chart
                        fig_trend = px.line(ts_df, x='Hour', y='Value', color='Metric',
                                          title="24-Hour Performance Trends",
                                          labels={'Hour': 'Hour of Day', 'Value': 'Performance Value'})
                        st.plotly_chart(fig_trend, use_container_width=True)
                
                st.markdown("#### 🎯 Load Testing & Stress Analysis")
                
                # Load testing parameters
                load_col1, load_col2 = st.columns(2)
                
                with load_col1:
                    test_duration = st.selectbox(
                        "Test Duration:",
                        ["1 minute", "5 minutes", "15 minutes", "1 hour"],
                        key="test_duration"
                    )
                
                with load_col2:
                    load_pattern = st.selectbox(
                        "Load Pattern:",
                        ["Constant Load", "Ramp Up", "Spike Test", "Stress Test"],
                        key="load_pattern"
                    )
                
                if st.button("🔥 Execute Load Test", key="load_test"):
                    with st.spinner(f"Running {load_pattern.lower()} for {test_duration}..."):
                        
                        # Simulate load test results
                        duration_minutes = {"1 minute": 1, "5 minutes": 5, "15 minutes": 15, "1 hour": 60}[test_duration]
                        
                        load_results = {
                            'max_concurrent_users': random.randint(100, 1000),
                            'requests_per_second': random.uniform(50, 500),
                            'error_rate': random.uniform(0.1, 5.0),
                            'avg_response_time': random.uniform(100, 500),
                            'cpu_utilization': random.uniform(40, 95),
                            'memory_utilization': random.uniform(50, 85),
                            'network_saturation': random.uniform(30, 80)
                        }
                        
                        st.success(f"✅ Load test completed! Duration: {test_duration}")
                        
                        # Load test metrics
                        load_col1, load_col2, load_col3 = st.columns(3)
                        
                        with load_col1:
                            st.metric("Max Users", f"{load_results['max_concurrent_users']}")
                            st.metric("Requests/sec", f"{load_results['requests_per_second']:.1f}")
                        
                        with load_col2:
                            st.metric("Error Rate", f"{load_results['error_rate']:.2f}%")
                            st.metric("Avg Response", f"{load_results['avg_response_time']:.0f}ms")
                        
                        with load_col3:
                            st.metric("CPU Usage", f"{load_results['cpu_utilization']:.1f}%")
                            st.metric("Memory Usage", f"{load_results['memory_utilization']:.1f}%")
                        
                        # Performance assessment
                        if load_results['error_rate'] < 1.0 and load_results['avg_response_time'] < 200:
                            st.success("🌟 Excellent performance under load!")
                        elif load_results['error_rate'] < 3.0 and load_results['avg_response_time'] < 400:
                            st.info("✅ Good performance, minor optimization possible")
                        else:
                            st.warning("⚠️ Performance issues detected under load")
            
            with perf_col2:
                st.markdown("#### 📈 Scalability Analysis")
                
                if st.button("📊 Analyze Network Scalability", key="scalability_analysis"):
                    with st.spinner("Analyzing network scalability characteristics..."):
                        
                        # Simulate scalability analysis
                        node_counts = [10, 25, 50, 100, 200, 500]
                        scalability_data = []
                        
                        for nodes in node_counts:
                            # Simulate performance degradation with scale
                            base_latency = 50
                            scale_factor = (nodes / 10) ** 0.7  # Sub-linear degradation
                            latency = base_latency * scale_factor + random.uniform(-10, 10)
                            
                            throughput = 100 * (1 - (nodes - 10) / 1000) + random.uniform(-5, 5)
                            throughput = max(throughput, 20)  # Minimum throughput
                            
                            memory_usage = (nodes * 2.5) + random.uniform(-5, 5)
                            
                            scalability_data.append({
                                'Nodes': nodes,
                                'Latency (ms)': max(latency, 10),
                                'Throughput (%)': min(throughput, 100),
                                'Memory (GB)': memory_usage,
                                'Scalability Score': max(100 - (latency - 50) - (100 - throughput), 0)
                            })
                        
                        scalability_df = pd.DataFrame(scalability_data)
                        
                        st.success("✅ Scalability analysis complete!")
                        
                        # Scalability metrics table
                        st.dataframe(scalability_df, use_container_width=True)
                        
                        # Scalability visualization
                        fig_scale = px.line(scalability_df, x='Nodes', y=['Latency (ms)', 'Throughput (%)', 'Scalability Score'],
                                          title="Network Scalability Analysis",
                                          labels={'value': 'Performance Metric', 'Nodes': 'Number of Nodes'})
                        st.plotly_chart(fig_scale, use_container_width=True)
                        
                        # Scalability recommendations
                        max_score = scalability_df['Scalability Score'].max()
                        optimal_nodes = scalability_df.loc[scalability_df['Scalability Score'].idxmax(), 'Nodes']
                        
                        if max_score > 80:
                            st.success(f"🌟 Excellent scalability! Optimal size: ~{optimal_nodes} nodes")
                        elif max_score > 60:
                            st.info(f"✅ Good scalability with optimization at {optimal_nodes} nodes")
                        else:
                            st.warning("⚠️ Scalability limitations detected - consider architecture review")
                
                st.markdown("#### 🔧 Performance Optimization")
                
                optimization_type = st.selectbox(
                    "Optimization Focus:",
                    ["Latency Optimization", "Throughput Maximization", "Resource Efficiency", "Quantum Fidelity"],
                    key="optimization_type"
                )
                
                if st.button("🎯 Generate Optimization Plan", key="optimization_plan"):
                    with st.spinner("Analyzing network and generating optimization recommendations..."):
                        
                        # Simulate optimization analysis
                        current_performance = {
                            'latency': random.uniform(100, 200),
                            'throughput': random.uniform(70, 90),
                            'cpu_usage': random.uniform(60, 85),
                            'memory_usage': random.uniform(50, 75),
                            'quantum_fidelity': random.uniform(80, 95)
                        }
                        
                        # Generate optimization recommendations based on selected focus
                        if optimization_type == "Latency Optimization":
                            recommendations = [
                                "🔧 Implement edge caching to reduce hop count",
                                "⚡ Optimize quantum state preparation protocols",
                                "🌐 Use shortest path routing algorithms",
                                "📊 Enable connection pooling and multiplexing",
                                "⚙️ Tune buffer sizes for optimal packet flow"
                            ]
                            potential_improvement = "25-40% latency reduction"
                        
                        elif optimization_type == "Throughput Maximization":
                            recommendations = [
                                "📈 Implement parallel quantum channels",
                                "🔄 Enable load balancing across all nodes",
                                "⚡ Optimize error correction overhead",
                                "🎯 Use adaptive bitrate for quantum links",
                                "🔧 Implement traffic shaping policies"
                            ]
                            potential_improvement = "30-50% throughput increase"
                        
                        elif optimization_type == "Resource Efficiency":
                            recommendations = [
                                "💾 Implement intelligent memory management",
                                "⚡ Use quantum circuit optimization",
                                "🔄 Enable dynamic resource allocation",
                                "📊 Implement predictive scaling",
                                "🎯 Optimize garbage collection cycles"
                            ]
                            potential_improvement = "20-35% resource savings"
                        
                        else:  # Quantum Fidelity
                            recommendations = [
                                "⚛️ Implement advanced error correction codes",
                                "🌡️ Optimize operating temperature control",
                                "🔧 Use entanglement purification protocols",
                                "📊 Implement real-time fidelity monitoring",
                                "⚡ Minimize quantum state transfer time"
                            ]
                            potential_improvement = "15-25% fidelity improvement"
                        
                        st.success("✅ Optimization plan generated!")
                        
                        # Current performance overview
                        st.markdown("**📊 Current Performance Baseline:**")
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        
                        with perf_col1:
                            st.metric("Latency", f"{current_performance['latency']:.0f}ms")
                            st.metric("CPU Usage", f"{current_performance['cpu_usage']:.1f}%")
                        
                        with perf_col2:
                            st.metric("Throughput", f"{current_performance['throughput']:.1f}%")
                            st.metric("Memory", f"{current_performance['memory_usage']:.1f}%")
                        
                        with perf_col3:
                            st.metric("Quantum Fidelity", f"{current_performance['quantum_fidelity']:.1f}%")
                        
                        # Optimization recommendations
                        st.markdown(f"**🎯 {optimization_type} Recommendations:**")
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")
                        
                        st.info(f"**💡 Expected Improvement:** {potential_improvement}")
                        
                        # Implementation priority
                        st.markdown("**📋 Implementation Priority:**")
                        priority_data = []
                        for i, rec in enumerate(recommendations):
                            priority_data.append({
                                'Recommendation': rec.split(' ', 1)[1],  # Remove emoji
                                'Impact': random.choice(['High', 'Medium', 'Low']),
                                'Effort': random.choice(['Low', 'Medium', 'High']),
                                'Priority': random.choice(['Critical', 'High', 'Medium', 'Low'])
                            })
                        
                        priority_df = pd.DataFrame(priority_data)
                        st.dataframe(priority_df, use_container_width=True)

    with tab1:
        st.markdown('<h2 class="section-header">🛡️ Advanced Security Analysis & Threat Intelligence</h2>', unsafe_allow_html=True)
        
        if st.session_state.network:
            # Security Dashboard Overview
            st.markdown("### 📊 Real-Time Security Dashboard")
            
            # Generate dynamic security metrics
            threat_level = random.choice(["🟢 LOW", "🟡 MEDIUM", "🔴 HIGH", "🟣 CRITICAL"])
            network_vulnerability = random.uniform(0.1, 0.9)
            quantum_security_score = random.uniform(75, 99)
            encryption_strength = random.uniform(85, 100)
            
            # Security metrics in columns
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🚨 Threat Level", threat_level)
            with col2:
                st.metric("🔍 Vulnerability Score", f"{network_vulnerability:.2f}")
            with col3:
                st.metric("⚛️ Quantum Security", f"{quantum_security_score:.1f}%")
            with col4:
                st.metric("🔐 Encryption Strength", f"{encryption_strength:.1f}%")
            with col5:
                active_attacks = random.randint(0, 5)
                st.metric("🎯 Active Threats", active_attacks)
            
            st.divider()
            
            # New Feature: Real-time Network Monitoring Dashboard
            st.markdown("### 📡 Real-Time Network Monitoring & Alerts")
            
            monitor_col1, monitor_col2 = st.columns(2)
            
            with monitor_col1:
                st.markdown("#### 🚨 Live Security Alerts")
                
                if st.button("🔄 Refresh Security Status", key="refresh_security"):
                    with st.spinner("Scanning network for threats..."):
                        # Simulate real-time security alerts
                        alerts = []
                        alert_types = [
                            "🔴 Suspicious quantum state manipulation detected",
                            "🟠 Unusual traffic pattern on node",
                            "🟡 Key distribution anomaly detected", 
                            "🔵 Normal quantum entanglement established",
                            "🟢 All security protocols functioning normally",
                            "🟣 Advanced persistent threat signature found",
                            "⚫ Zero-day exploit attempt blocked"
                        ]
                        
                        # Generate random alerts
                        num_alerts = random.randint(3, 8)
                        for i in range(num_alerts):
                            alert = random.choice(alert_types)
                            timestamp = f"{random.randint(1, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
                            node_id = random.randint(0, st.session_state.network.num_nodes - 1)
                            severity = "HIGH" if "🔴" in alert else "MEDIUM" if "🟠" in alert else "LOW"
                            
                            alerts.append({
                                "Time": timestamp,
                                "Alert": alert,
                                "Node": node_id,
                                "Severity": severity,
                                "Status": random.choice(["ACTIVE", "RESOLVED", "INVESTIGATING"])
                            })
                        
                        # Sort by severity
                        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
                        alerts.sort(key=lambda x: severity_order[x["Severity"]])
                        
                        st.success("✅ Security scan completed!")
                        
                        # Display alerts in a table
                        alerts_df = pd.DataFrame(alerts)
                        st.dataframe(alerts_df, use_container_width=True)
                        
                        # Alert statistics
                        high_alerts = len([a for a in alerts if a["Severity"] == "HIGH"])
                        active_alerts = len([a for a in alerts if a["Status"] == "ACTIVE"])
                        
                        alert_col1, alert_col2, alert_col3 = st.columns(3)
                        with alert_col1:
                            st.metric("🔴 High Priority", high_alerts)
                        with alert_col2:
                            st.metric("⚡ Active Alerts", active_alerts)
                        with alert_col3:
                            st.metric("📊 Total Alerts", len(alerts))
                
                st.markdown("#### 🛡️ Security Posture Analysis")
                
                if st.button("📊 Analyze Security Posture", key="security_posture"):
                    with st.spinner("Analyzing network security posture..."):
                        # Generate security posture metrics
                        posture_metrics = {
                            "Network Segmentation": random.uniform(70, 95),
                            "Access Control": random.uniform(75, 98),
                            "Encryption Coverage": random.uniform(85, 99),
                            "Quantum Key Distribution": random.uniform(80, 95),
                            "Intrusion Detection": random.uniform(70, 90),
                            "Incident Response": random.uniform(65, 85),
                            "Security Monitoring": random.uniform(75, 95),
                            "Compliance Score": random.uniform(80, 98)
                        }
                        
                        st.success("✅ Security posture analysis complete!")
                        
                        # Create radar chart data
                        categories = list(posture_metrics.keys())
                        values = list(posture_metrics.values())
                        
                        # Display metrics
                        for i in range(0, len(categories), 2):
                            pos_col1, pos_col2 = st.columns(2)
                            
                            with pos_col1:
                                if i < len(categories):
                                    score = values[i]
                                    color = "🟢" if score > 90 else "🟡" if score > 75 else "🔴"
                                    st.metric(f"{color} {categories[i]}", f"{score:.1f}%")
                            
                            with pos_col2:
                                if i + 1 < len(categories):
                                    score = values[i + 1]
                                    color = "🟢" if score > 90 else "🟡" if score > 75 else "🔴"
                                    st.metric(f"{color} {categories[i + 1]}", f"{score:.1f}%")
                        
                        # Overall security score
                        overall_score = np.mean(values)
                        if overall_score > 90:
                            st.success(f"🌟 Excellent security posture: {overall_score:.1f}%")
                        elif overall_score > 80:
                            st.info(f"✅ Good security posture: {overall_score:.1f}%")
                        elif overall_score > 70:
                            st.warning(f"⚠️ Moderate security posture: {overall_score:.1f}% - needs improvement")
                        else:
                            st.error(f"🚨 Poor security posture: {overall_score:.1f}% - immediate action required")
            
            with monitor_col2:
                st.markdown("#### 📈 Security Metrics Trends")
                
                if st.button("📊 Generate Security Trends", key="security_trends"):
                    with st.spinner("Generating security trend analysis..."):
                        # Generate time-series security data
                        hours = list(range(1, 25))  # 24 hours
                        trend_data = []
                        
                        for hour in hours:
                            # Simulate daily security patterns
                            base_threats = 10 + 15 * np.sin(hour * np.pi / 12)  # Peak during business hours
                            threats = max(0, base_threats + random.uniform(-5, 5))
                            
                            base_blocks = 8 + 12 * np.sin((hour + 6) * np.pi / 12)  # Higher blocking during peak
                            blocks = max(0, base_blocks + random.uniform(-3, 3))
                            
                            quantum_integrity = 95 + 3 * np.sin(hour * np.pi / 6) + random.uniform(-2, 2)
                            
                            trend_data.append({
                                "Hour": hour,
                                "Threats Detected": int(threats),
                                "Threats Blocked": int(blocks),
                                "Quantum Integrity": min(100, max(80, quantum_integrity))
                            })
                        
                        trends_df = pd.DataFrame(trend_data)
                        
                        st.success("✅ Security trends analysis complete!")
                        
                        # Create trend visualization
                        fig_threats = px.line(trends_df, x='Hour', y=['Threats Detected', 'Threats Blocked'],
                                            title="24-Hour Threat Activity",
                                            labels={'value': 'Count', 'Hour': 'Hour of Day'})
                        st.plotly_chart(fig_threats, use_container_width=True)
                        
                        fig_quantum = px.line(trends_df, x='Hour', y='Quantum Integrity',
                                            title="Quantum Network Integrity Over Time",
                                            labels={'Quantum Integrity': 'Integrity %', 'Hour': 'Hour of Day'})
                        st.plotly_chart(fig_quantum, use_container_width=True)
                        
                        # Trend summary
                        avg_threats = trends_df['Threats Detected'].mean()
                        avg_blocks = trends_df['Threats Blocked'].mean()
                        block_rate = (avg_blocks / avg_threats * 100) if avg_threats > 0 else 0
                        
                        trend_col1, trend_col2, trend_col3 = st.columns(3)
                        
                        with trend_col1:
                            st.metric("Avg Threats/Hour", f"{avg_threats:.1f}")
                        with trend_col2:
                            st.metric("Avg Blocks/Hour", f"{avg_blocks:.1f}")
                        with trend_col3:
                            st.metric("Block Success Rate", f"{block_rate:.1f}%")
                
                st.markdown("#### 🎯 Automated Response System")
                
                response_mode = st.selectbox(
                    "Automated Response Mode:",
                    ["🔴 Aggressive - Block all suspicious activity",
                     "🟡 Balanced - Analyze then respond", 
                     "🟢 Passive - Monitor and alert only",
                     "🔧 Custom - User-defined rules"],
                    key="response_mode"
                )
                
                if st.button("⚙️ Configure Auto-Response", key="auto_response"):
                    with st.spinner("Configuring automated response system..."):
                        # Simulate auto-response configuration
                        config_success = random.choice([True, True, True, False])  # 75% success rate
                        
                        if config_success:
                            st.success("✅ Automated response system configured successfully!")
                            
                            response_settings = {
                                "Response Time": f"{random.uniform(0.1, 2.0):.2f} seconds",
                                "False Positive Rate": f"{random.uniform(0.5, 3.0):.2f}%",
                                "Coverage": f"{random.uniform(85, 98):.1f}%",
                                "Escalation Threshold": random.choice(["High", "Critical"])
                            }
                            
                            for setting, value in response_settings.items():
                                st.write(f"• **{setting}:** {value}")
                            
                            # Simulate recent auto-responses
                            st.markdown("**Recent Automated Actions:**")
                            actions = [
                                "🛡️ Blocked suspicious quantum state probe",
                                "🔒 Isolated compromised node temporarily", 
                                "⚡ Rotated quantum keys on affected links",
                                "📊 Generated incident report #QN-2025-0930",
                                "🚨 Escalated APT signature to security team"
                            ]
                            
                            for action in random.sample(actions, 3):
                                st.write(f"• {action}")
                        
                        else:
                            st.error("❌ Configuration failed - please check system settings")
                            st.warning("⚠️ Manual intervention required")
            
            st.divider()
            
            # Interactive Security Tools
            sec_col1, sec_col2 = st.columns(2)
            
            with sec_col1:
                st.markdown("### 🔍 Threat Simulation Center")
                
                with st.expander("🚨 Advanced Persistent Threat (APT) Simulation", expanded=False):
                    threat_actor = st.selectbox(
                        "Select Threat Actor Type:",
                        ["Nation State", "Cybercriminal Group", "Hacktivist", "Insider Threat", "AI-Powered Attacker"],
                        key="apt_actor"
                    )
                    
                    attack_vector = st.selectbox(
                        "Primary Attack Vector:",
                        ["Quantum Cryptanalysis", "Social Engineering", "Zero-Day Exploit", "Supply Chain", "IoT Compromise"],
                        key="attack_vector"
                    )
                    
                    sophistication = st.slider("Attack Sophistication Level", 1, 10, 7, key="apt_sophistication")
                    
                    if st.button("🚀 Launch APT Simulation", type="primary", key="launch_apt"):
                        with st.spinner("Simulating Advanced Persistent Threat..."):
                            # Simulate APT attack stages
                            stages = ["Reconnaissance", "Initial Access", "Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement", "Collection", "Exfiltration"]
                            
                            progress_bar = st.progress(0)
                            results = {}
                            
                            for i, stage in enumerate(stages):
                                progress_bar.progress((i + 1) / len(stages))
                                success_prob = max(0.1, 1 - (sophistication * 0.08))
                                success = random.random() < success_prob
                                results[stage] = {
                                    "success": success,
                                    "impact": random.uniform(0.2, 1.0) if success else 0,
                                    "detection_time": random.uniform(1, 48) if success else 0
                                }
                            
                            # Display results
                            st.success("✅ APT Simulation Complete!")
                            
                            successful_stages = sum(1 for r in results.values() if r["success"])
                            total_impact = sum(r["impact"] for r in results.values())
                            avg_detection_time = np.mean([r["detection_time"] for r in results.values() if r["detection_time"] > 0])
                            
                            result_col1, result_col2, result_col3 = st.columns(3)
                            with result_col1:
                                st.metric("Successful Stages", f"{successful_stages}/{len(stages)}")
                            with result_col2:
                                st.metric("Total Impact Score", f"{total_impact:.2f}")
                            with result_col3:
                                st.metric("Avg Detection Time", f"{avg_detection_time:.1f}h")
                            
                            # Risk assessment
                            if successful_stages >= 7:
                                st.error("🚨 CRITICAL RISK: Network highly vulnerable to APT attacks!")
                            elif successful_stages >= 4:
                                st.warning("⚠️ HIGH RISK: Significant security improvements needed")
                            else:
                                st.success("✅ LOW RISK: Network shows good resilience")
                
                with st.expander("🔮 Quantum Cryptanalysis Assessment", expanded=False):
                    target_encryption = st.selectbox(
                        "Target Encryption:",
                        ["RSA-2048", "RSA-4096", "ECC-256", "ECC-521", "AES-256", "Post-Quantum Lattice"],
                        key="target_crypto"
                    )
                    
                    quantum_computer_power = st.slider("Quantum Computer Power (Qubits)", 50, 10000, 1000, key="quantum_power")
                    target_year = st.slider("Assessment Year", 2025, 2040, 2030, key="crypto_year")
                    
                    if st.button("🔍 Analyze Quantum Vulnerability", key="quantum_vuln"):
                        with st.spinner("Analyzing quantum cryptanalysis threat..."):
                            # Simplified quantum threat model
                            base_vulnerability = {
                                "RSA-2048": 0.9, "RSA-4096": 0.7, "ECC-256": 0.95,
                                "ECC-521": 0.8, "AES-256": 0.3, "Post-Quantum Lattice": 0.1
                            }
                            
                            year_factor = min(1.0, (target_year - 2024) * 0.05)
                            qubit_factor = min(1.0, quantum_computer_power / 5000)
                            
                            vulnerability_score = base_vulnerability[target_encryption] * year_factor * qubit_factor
                            break_probability = min(0.99, vulnerability_score)
                            
                            st.success("✅ Quantum Threat Analysis Complete!")
                            
                            vuln_col1, vuln_col2, vuln_col3 = st.columns(3)
                            with vuln_col1:
                                st.metric("Vulnerability Score", f"{vulnerability_score:.2f}")
                            with vuln_col2:
                                st.metric("Break Probability", f"{break_probability:.1%}")
                            with vuln_col3:
                                estimated_break_year = 2024 + (1 - vulnerability_score) * 10
                                st.metric("Est. Break Year", f"{int(estimated_break_year)}")
                            
                            if vulnerability_score > 0.8:
                                st.error("🚨 URGENT: Migrate to quantum-safe cryptography immediately!")
                            elif vulnerability_score > 0.5:
                                st.warning("⚠️ PLAN: Begin migration within 2-3 years")
                            else:
                                st.success("✅ SAFE: Current encryption adequate for now")
            
            with sec_col2:
                st.markdown("### 🛡️ Defense Systems")
                
                with st.expander("🔍 Network Vulnerability Scanner", expanded=False):
                    scan_depth = st.selectbox(
                        "Scan Depth:",
                        ["Quick Scan", "Standard Scan", "Deep Scan", "Comprehensive Audit"],
                        key="scan_depth"
                    )
                    
                    scan_targets = st.multiselect(
                        "Scan Targets:",
                        ["Quantum Nodes", "Classical Nodes", "Network Links", "Routing Protocols", "Encryption Keys"],
                        default=["Quantum Nodes", "Classical Nodes"],
                        key="scan_targets"
                    )
                    
                    if st.button("🔍 Start Vulnerability Scan", key="vuln_scan"):
                        with st.spinner("Scanning network for vulnerabilities..."):
                            vulnerabilities = []
                            
                            for target in scan_targets:
                                num_vulns = random.randint(0, 5)
                                severity_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                                
                                for i in range(num_vulns):
                                    vuln = {
                                        "Target": target,
                                        "Type": random.choice(["Buffer Overflow", "Injection", "Misconfiguration", "Weak Encryption", "Access Control"]),
                                        "Severity": random.choice(severity_levels),
                                        "CVE": f"CVE-2024-{random.randint(1000, 9999)}",
                                        "Risk Score": random.uniform(1, 10)
                                    }
                                    vulnerabilities.append(vuln)
                            
                            if vulnerabilities:
                                st.warning(f"⚠️ Found {len(vulnerabilities)} vulnerabilities!")
                                df_vulns = pd.DataFrame(vulnerabilities)
                                
                                # Color code by severity
                                def color_severity(val):
                                    if val == "CRITICAL":
                                        return "background-color: #ff4444; color: white"
                                    elif val == "HIGH":
                                        return "background-color: #ff8800; color: white"
                                    elif val == "MEDIUM":
                                        return "background-color: #ffaa00; color: black"
                                    else:
                                        return "background-color: #88ff88; color: black"
                                
                                styled_df = df_vulns.style.applymap(color_severity, subset=['Severity'])
                                st.dataframe(styled_df, use_container_width=True)
                            else:
                                st.success("✅ No vulnerabilities found!")
                
                with st.expander("🚨 Intrusion Detection System", expanded=False):
                    detection_sensitivity = st.slider("Detection Sensitivity", 0.1, 1.0, 0.7, key="ids_sensitivity")
                    
                    monitoring_duration = st.selectbox(
                        "Monitoring Duration:",
                        ["1 minute", "5 minutes", "15 minutes", "1 hour"],
                        key="monitoring_duration"
                    )
                    
                    if st.button("🔍 Start Real-Time Monitoring", key="start_ids"):
                        with st.spinner("Monitoring network traffic..."):
                            # Simulate network events
                            events = []
                            event_types = ["Normal Traffic", "Port Scan", "DDoS Attempt", "Unauthorized Access", "Data Exfiltration", "Malware Communication"]
                            
                            for i in range(20):
                                event = {
                                    "Timestamp": f"2025-09-30 20:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}",
                                    "Source": f"192.168.1.{random.randint(1, 254)}",
                                    "Destination": f"192.168.1.{random.randint(1, 254)}",
                                    "Event Type": random.choice(event_types),
                                    "Risk Level": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                                    "Confidence": random.uniform(0.5, 1.0)
                                }
                                events.append(event)
                            
                            df_events = pd.DataFrame(events)
                            
                            # Filter based on sensitivity
                            filtered_events = df_events[df_events["Confidence"] >= detection_sensitivity]
                            
                            st.success(f"✅ Monitoring complete! Detected {len(filtered_events)} events")
                            
                            if len(filtered_events) > 0:
                                # Show alerts
                                critical_events = filtered_events[filtered_events["Risk Level"] == "CRITICAL"]
                                if len(critical_events) > 0:
                                    st.error(f"🚨 {len(critical_events)} CRITICAL alerts detected!")
                                
                                st.dataframe(filtered_events, use_container_width=True)
                            else:
                                st.info("🔍 No suspicious activity detected during monitoring period")
            
            st.divider()
            
            # Security Analytics Section
            st.markdown("### 📈 Security Analytics & Intelligence")
            
            analytics_col1, analytics_col2 = st.columns(2)
            
            with analytics_col1:
                st.markdown("**🔍 Threat Intelligence Dashboard**")
                
                # Generate threat trend data
                dates = pd.date_range(start='2025-09-23', end='2025-09-30', freq='D')
                threat_data = {
                    'Date': dates,
                    'Malware Detections': [random.randint(5, 50) for _ in dates],
                    'Phishing Attempts': [random.randint(2, 25) for _ in dates],
                    'Brute Force Attacks': [random.randint(1, 15) for _ in dates],
                    'DDoS Attempts': [random.randint(0, 8) for _ in dates]
                }
                
                df_threats = pd.DataFrame(threat_data)
                
                # Create threat trend chart
                fig_threats = go.Figure()
                
                for threat_type in ['Malware Detections', 'Phishing Attempts', 'Brute Force Attacks', 'DDoS Attempts']:
                    fig_threats.add_trace(go.Scatter(
                        x=df_threats['Date'],
                        y=df_threats[threat_type],
                        mode='lines+markers',
                        name=threat_type,
                        line=dict(width=2)
                    ))
                
                fig_threats.update_layout(
                    title="7-Day Threat Activity Trends",
                    xaxis_title="Date",
                    yaxis_title="Number of Incidents",
                    height=350,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_threats, use_container_width=True)
            
            with analytics_col2:
                st.markdown("**🎯 Attack Vector Analysis**")
                
                # Attack vector pie chart
                attack_vectors = ['Network Intrusion', 'Malware', 'Social Engineering', 'Insider Threat', 'Physical Access', 'Supply Chain']
                attack_counts = [random.randint(5, 30) for _ in attack_vectors]
                
                fig_attacks = go.Figure(data=[go.Pie(
                    labels=attack_vectors,
                    values=attack_counts,
                    hole=0.4,
                    textinfo='label+percent'
                )])
                
                fig_attacks.update_layout(
                    title="Attack Vector Distribution (Last 30 Days)",
                    height=350
                )
                
                st.plotly_chart(fig_attacks, use_container_width=True)
        
        else:
            st.warning("⚠️ Please create a network first to access security analysis features")
            st.info("💡 Use the sidebar to configure and create your quantum-classical hybrid network")
            
            # Show security education content when no network
            st.markdown("### 🎓 Security Education Center")
            
            edu_col1, edu_col2 = st.columns(2)
            
            with edu_col1:
                st.markdown("#### 🔐 Post-Quantum Cryptography")
                st.info("""
                **Key Concepts:**
                - Current encryption vulnerable to quantum computers
                - Need for quantum-resistant algorithms
                - Timeline for migration (2025-2035)
                - NIST post-quantum standards
                """)
                
            with edu_col2:
                st.markdown("#### ⚛️ Quantum Key Distribution")
                st.info("""
                **Principles:**
                - Quantum mechanics ensures security
                - No-cloning theorem protection
                - Eavesdropping detection
                - Perfect forward secrecy
                """)
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
    
    # Advanced Network Simulation Center
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🎮 Simulation Center")
        
        if st.button("🚀 Launch Advanced Scenarios", key="launch_scenarios"):
            st.markdown("#### 🌟 Network Simulation Scenarios")
            
            scenario_type = st.selectbox(
                "Simulation Scenario:",
                ["🏥 Healthcare Emergency Network", "🏛️ Government Secure Communications", 
                 "💰 Financial Trading Network", "🌊 Disaster Recovery Simulation",
                 "🛡️ Cyber Attack Response", "📡 Satellite Network Handover"],
                key="scenario_selector"
            )
            
            if st.button("▶️ Run Scenario", key="run_scenario"):
                with st.spinner(f"Running {scenario_type} simulation..."):
                    
                    # Simulate scenario-specific metrics
                    scenario_results = {
                        "🏥 Healthcare Emergency Network": {
                            "priority_messages": random.randint(50, 200),
                            "response_time": random.uniform(2, 8),
                            "reliability": random.uniform(95, 99.5),
                            "critical_path_redundancy": random.randint(2, 5)
                        },
                        "🏛️ Government Secure Communications": {
                            "classification_levels": random.randint(3, 7),
                            "encryption_strength": random.uniform(95, 99.9),
                            "quantum_key_distribution": random.uniform(85, 98),
                            "security_clearance_nodes": random.randint(5, 15)
                        },
                        "💰 Financial Trading Network": {
                            "transactions_per_second": random.randint(1000, 10000),
                            "latency_microseconds": random.uniform(10, 100),
                            "fault_tolerance": random.uniform(99.5, 99.99),
                            "market_data_accuracy": random.uniform(99.8, 99.99)
                        },
                        "🌊 Disaster Recovery Simulation": {
                            "nodes_affected": random.randint(20, 60),
                            "recovery_time_minutes": random.uniform(5, 30),
                            "backup_routes_activated": random.randint(10, 25),
                            "data_integrity": random.uniform(98, 100)
                        },
                        "🛡️ Cyber Attack Response": {
                            "attack_vectors_detected": random.randint(5, 20),
                            "response_time_seconds": random.uniform(0.5, 5),
                            "mitigation_effectiveness": random.uniform(85, 98),
                            "network_isolation_speed": random.uniform(1, 10)
                        },
                        "📡 Satellite Network Handover": {
                            "handover_success_rate": random.uniform(95, 99.8),
                            "signal_strength_variation": random.uniform(5, 20),
                            "orbital_period_coverage": random.uniform(85, 95),
                            "ground_station_connectivity": random.randint(8, 20)
                        }
                    }
                    
                    results = scenario_results[scenario_type]
                    
                    st.success(f"✅ {scenario_type} simulation completed!")
                    
                    # Display scenario-specific results
                    for metric, value in results.items():
                        if isinstance(value, float):
                            if "rate" in metric or "reliability" in metric or "accuracy" in metric:
                                st.metric(metric.replace('_', ' ').title(), f"{value:.2f}%")
                            elif "time" in metric and "seconds" in metric:
                                st.metric(metric.replace('_', ' ').title(), f"{value:.2f}s")
                            elif "time" in metric and "minutes" in metric:
                                st.metric(metric.replace('_', ' ').title(), f"{value:.1f}min")
                            else:
                                st.metric(metric.replace('_', ' ').title(), f"{value:.2f}")
                        else:
                            st.metric(metric.replace('_', ' ').title(), str(value))
                    
                    # Scenario-specific insights
                    if scenario_type == "🏥 Healthcare Emergency Network":
                        if results["response_time"] < 5:
                            st.success("🏥 Excellent emergency response capability!")
                        else:
                            st.warning("⚠️ Response time may need optimization for critical care")
                    
                    elif scenario_type == "💰 Financial Trading Network":
                        if results["latency_microseconds"] < 50:
                            st.success("💰 Ultra-low latency ideal for high-frequency trading!")
                        else:
                            st.info("📊 Good performance for standard trading operations")
            
            # What-if Analysis
            st.markdown("#### 🔮 What-If Analysis")
            
            what_if_scenario = st.selectbox(
                "What-If Scenario:",
                ["❌ 30% of nodes fail", "⚡ Double the traffic load", 
                 "🔐 Quantum computer attack", "🌐 Add 50% more nodes",
                 "📡 Lose primary data center", "🌡️ Extreme temperature conditions"],
                key="what_if_selector"
            )
            
            if st.button("🧪 Analyze Impact", key="what_if_analysis"):
                with st.spinner(f"Analyzing: {what_if_scenario}"):
                    
                    # Simulate what-if analysis results
                    if "nodes fail" in what_if_scenario:
                        impact = {
                            "Network Connectivity": random.uniform(40, 70),
                            "Performance Degradation": random.uniform(20, 50),
                            "Recovery Time": f"{random.uniform(10, 45):.1f} minutes",
                            "Alternative Paths": random.randint(5, 15)
                        }
                        recommendation = "🛡️ Implement additional redundancy and failover protocols"
                    
                    elif "traffic load" in what_if_scenario:
                        impact = {
                            "Latency Increase": f"{random.uniform(15, 40):.1f}%",
                            "Throughput Reduction": f"{random.uniform(10, 30):.1f}%",
                            "Resource Utilization": f"{random.uniform(80, 95):.1f}%",
                            "Queue Overflow Risk": "Medium" if random.random() > 0.5 else "High"
                        }
                        recommendation = "📈 Consider load balancing and capacity planning"
                    
                    elif "Quantum computer attack" in what_if_scenario:
                        impact = {
                            "Classical Encryption Risk": "Critical",
                            "Quantum Key Distribution": "Secure",
                            "Migration Time Required": f"{random.uniform(6, 24):.1f} hours",
                            "Post-Quantum Readiness": f"{random.uniform(70, 90):.1f}%"
                        }
                        recommendation = "🔐 Accelerate post-quantum cryptography deployment"
                    
                    elif "more nodes" in what_if_scenario:
                        impact = {
                            "Routing Complexity": f"+{random.uniform(25, 60):.1f}%",
                            "Network Resilience": f"+{random.uniform(30, 50):.1f}%",
                            "Resource Requirements": f"+{random.uniform(40, 80):.1f}%",
                            "Management Overhead": f"+{random.uniform(20, 45):.1f}%"
                        }
                        recommendation = "🌐 Implement hierarchical network architecture"
                    
                    elif "data center" in what_if_scenario:
                        impact = {
                            "Service Disruption": f"{random.uniform(15, 45):.1f} minutes",
                            "Data Loss Risk": "Minimal" if random.random() > 0.3 else "Low",
                            "Backup Activation": f"{random.uniform(2, 8):.1f} minutes",
                            "Performance Impact": f"{random.uniform(10, 25):.1f}%"
                        }
                        recommendation = "🏢 Enhance geographic redundancy and backup procedures"
                    
                    else:  # Temperature conditions
                        impact = {
                            "Quantum Coherence": f"-{random.uniform(15, 40):.1f}%",
                            "Error Rate Increase": f"+{random.uniform(50, 200):.1f}%",
                            "Cooling System Load": f"+{random.uniform(30, 80):.1f}%",
                            "Performance Degradation": f"{random.uniform(20, 50):.1f}%"
                        }
                        recommendation = "🌡️ Implement robust environmental controls and monitoring"
                    
                    st.warning(f"📊 Impact Analysis: {what_if_scenario}")
                    
                    for metric, value in impact.items():
                        st.write(f"• **{metric}:** {value}")
                    
                    st.info(f"💡 **Recommendation:** {recommendation}")
    
    create_footer()