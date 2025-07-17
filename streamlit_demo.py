"""
Streamlit demo for quantum network simulation.
Run with: streamlit run streamlit_demo.py
"""
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os
import tempfile
from io import BytesIO

# Import our modules
from config import ConfigManager
from network import HybridNetworkSimulator
from link_sim import QuantumLinkSimulator
from routing import HybridRoutingProtocol
from pki import SymmetricKeyPKI
from repeater import QuantumRepeater

# Configure page - MUST be first Streamlit command
st.set_page_config(
    page_title="Quantum Network Simulation",
    page_icon="🔬",
    layout="wide"
)

# Add tabs
tab1, tab2 = st.tabs(["🔬 Simulation", "📚 Theory"])


def main():
    """Main Streamlit application."""
    st.title("🔬 Quantum-Classical Hybrid Network Simulation")
    st.markdown("Interactive demonstration of quantum networking principles")
    
    # Sidebar for parameters
    st.sidebar.header("Simulation Parameters")
    
    # Network parameters
    st.sidebar.subheader("Network Configuration")
    num_nodes = st.sidebar.slider("Number of Nodes", 5, 50, 12)
    quantum_ratio = st.sidebar.slider("Quantum Node Ratio", 0.1, 0.9, 0.4)
    prob_edge = st.sidebar.slider("Edge Probability", 0.1, 0.8, 0.3)
    
    # Simulation parameters
    st.sidebar.subheader("Physics Parameters")
    decoherence_rate = st.sidebar.slider("Decoherence Rate (/km)", 0.001, 0.1, 0.01)
    entanglement_base = st.sidebar.slider("Base Entanglement Success", 0.5, 0.95, 0.8)
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Settings")
    num_simulations = st.sidebar.slider("Number of Simulations", 100, 2000, 500)
    
    # Repeater parameters
    st.sidebar.subheader("Quantum Repeater Settings")
    enable_repeaters = st.sidebar.checkbox("Enable Quantum Repeaters", value=True)
    num_repeaters = st.sidebar.slider("Number of Repeaters", 1, 5, 3, disabled=not enable_repeaters)
    repeater_efficiency = st.sidebar.slider("Repeater Efficiency", 0.5, 0.95, 0.85, disabled=not enable_repeaters)
    repeater_tests = st.sidebar.slider("Repeater Tests", 50, 500, 100, disabled=not enable_repeaters)
    
    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        run_simulation(num_nodes, quantum_ratio, prob_edge, 
                      decoherence_rate, entanglement_base, num_simulations,
                      enable_repeaters, num_repeaters, repeater_efficiency, repeater_tests)


def run_simulation(num_nodes, quantum_ratio, prob_edge, 
                  decoherence_rate, entanglement_base, num_simulations,
                  enable_repeaters, num_repeaters, repeater_efficiency, repeater_tests):
    """Run the simulation with given parameters."""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Create network
        status_text.text("Creating network topology...")
        progress_bar.progress(10)
        
        network = HybridNetworkSimulator(
            num_nodes=num_nodes,
            quantum_node_ratio=quantum_ratio,
            prob_edge=prob_edge
        )
        
        # Step 2: Network visualization
        status_text.text("Generating network visualization...")
        progress_bar.progress(30)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Network Topology")
            
            # Create network plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Use the existing network visualization but capture the figure
            network.node_positions = nx.spring_layout(network.G, k=2, iterations=50)
            
            # Draw quantum nodes
            nx.draw_networkx_nodes(
                network.G, network.node_positions,
                nodelist=list(network.quantum_nodes),
                node_color='red', node_size=700,
                node_shape='s', label='Quantum Nodes',
                ax=ax
            )
            
            # Draw classical nodes
            nx.draw_networkx_nodes(
                network.G, network.node_positions,
                nodelist=list(network.classical_nodes),
                node_color='blue', node_size=700,
                node_shape='o', label='Classical Nodes',
                ax=ax
            )
            
            # Draw edges
            quantum_edges = [(u, v) for u, v in network.G.edges() 
                           if network.link_properties[(u, v)]['type'] == 'quantum']
            classical_edges = [(u, v) for u, v in network.G.edges() 
                             if network.link_properties[(u, v)]['type'] == 'classical']
            
            nx.draw_networkx_edges(
                network.G, network.node_positions,
                edgelist=quantum_edges,
                edge_color='red', style='dashed', width=2,
                ax=ax
            )
            nx.draw_networkx_edges(
                network.G, network.node_positions,
                edgelist=classical_edges,
                edge_color='blue', width=1,
                ax=ax
            )
            
            # Draw labels
            nx.draw_networkx_labels(network.G, network.node_positions, ax=ax)
            
            plt.title(f"Hybrid Network ({num_nodes} nodes)")
            plt.legend()
            plt.axis('off')
            
            st.pyplot(plt.gcf())
            plt.close()
        
        with col2:
            st.subheader("Network Statistics")
            stats = network.get_network_stats()
            
            # Display stats in a nice format
            st.metric("Total Nodes", stats['total_nodes'])
            st.metric("Quantum Nodes", stats['quantum_nodes'], 
                     f"{stats['quantum_nodes']/stats['total_nodes']*100:.1f}%")
            st.metric("Classical Nodes", stats['classical_nodes'],
                     f"{stats['classical_nodes']/stats['total_nodes']*100:.1f}%")
            st.metric("Total Edges", stats['total_edges'])
            st.metric("Average Degree", f"{stats['average_degree']:.2f}")
            
            if stats['diameter']:
                st.metric("Network Diameter", stats['diameter'])
        
        # Step 3: Link behavior analysis
        status_text.text("Analyzing link behavior...")
        progress_bar.progress(50)
        
        link_sim = QuantumLinkSimulator(
            network,
            decoherence_rate=decoherence_rate,
            entanglement_success_base=entanglement_base
        )
        
        link_results = link_sim.analyze_link_behavior(num_simulations=num_simulations)
        
        st.subheader("Link Performance Analysis")
        
        # Success rates
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Quantum Success Rate", 
                     f"{link_results['quantum_success_rate']:.2%}")
        with col2:
            st.metric("Classical Success Rate", 
                     f"{link_results['classical_success_rate']:.2%}")
        
        # Latency metrics
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Quantum Avg Latency", 
                     f"{link_results['quantum_avg_latency']:.2f}ms")
        with col4:
            st.metric("Classical Avg Latency", 
                     f"{link_results['classical_avg_latency']:.2f}ms")
        
        # Detailed Physics Effects Breakdown
        st.subheader("🔬 Detailed Physics Effects")
        
        # Individual failure causes
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Decoherence Failures", 
                     f"{link_results['decoherence_failure_rate']:.2%}",
                     help="Rate of quantum transmission failures due to decoherence")
        with col6:
            st.metric("Entanglement Failures", 
                     f"{link_results['entanglement_failure_rate']:.2%}",
                     help="Rate of failures in entanglement distribution")
        with col7:
            st.metric("Packet Loss Failures", 
                     f"{link_results['packet_loss_failure_rate']:.2%}",
                     help="Rate of classical transmission packet loss")
        with col8:
            st.metric("No-Cloning Violations", 
                     f"{link_results['no_cloning_violation_rate']:.2%}",
                     help="Rate of quantum no-cloning theorem violations detected")
        
        # Effect probabilities
        st.subheader("📊 Average Effect Probabilities")
        col9, col10, col11 = st.columns(3)
        with col9:
            st.metric("Avg Decoherence Prob", 
                     f"{link_results['avg_decoherence_prob']:.2%}",
                     help="Average probability of decoherence based on distance")
        with col10:
            st.metric("Avg Entanglement Success", 
                     f"{link_results['avg_entanglement_success_prob']:.2%}",
                     help="Average probability of successful entanglement distribution")
        with col11:
            st.metric("Avg Packet Loss Prob", 
                     f"{link_results['avg_packet_loss_prob']:.2%}",
                     help="Average probability of classical packet loss")
        
        # Step 4: PKI Analysis
        status_text.text("Analyzing PKI approaches...")
        progress_bar.progress(80)
        
        st.subheader("PKI Comparison")
        
        pki = SymmetricKeyPKI(num_users=25)
        
        # Get comparison data
        pairwise = pki.option1_pairwise_keys()
        kdc = pki.option2_kdc()
        hierarchical = pki.option3_group_key_hierarchy()
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Method': ['Pairwise', 'KDC', 'Hierarchical'],
            'Total Keys': [pairwise['total_keys'], kdc['total_keys'], hierarchical['total_keys']],
            'Keys per User': [pairwise['keys_per_user'], kdc['keys_per_user'], hierarchical['keys_per_user']],
            'Scalability': [pairwise['scalability'], kdc['scalability'], hierarchical['scalability']]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # PKI visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = comparison_df['Method']
        total_keys = comparison_df['Total Keys']
        
        bars = ax.bar(methods, total_keys, color=['red', 'green', 'orange'])
        ax.set_ylabel('Total Keys Required')
        ax.set_title('PKI Key Requirements Comparison')
        
        # Add value labels
        for bar, keys in zip(bars, total_keys):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{keys}', ha='center', va='bottom')
        
        st.pyplot(fig)
        plt.close()
        
        # Step 5: Quantum Repeater Analysis
        if enable_repeaters:
            status_text.text("Testing quantum repeaters...")
            progress_bar.progress(90)
            
            st.subheader("🔄 Quantum Repeater Performance")
            
            # Create repeater system
            repeater_system = QuantumRepeater(network, repeater_efficiency)
            repeater_system.place_repeaters(num_repeaters, strategy='centrality')
            
            # Run comparison
            with st.spinner("Comparing performance with and without repeaters..."):
                performance = repeater_system.compare_performance(num_tests=repeater_tests)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Without Repeaters", 
                         f"{performance['without_repeaters']:.2%}",
                         help="Baseline quantum transmission success rate")
            
            with col2:
                st.metric("With Repeaters", 
                         f"{performance['with_repeaters']:.2%}",
                         help="Enhanced transmission success rate with repeaters")
            
            with col3:
                improvement = performance['improvement']
                st.metric("Performance Improvement", 
                         f"{improvement:.1f}%",
                         delta=f"{improvement:.1f}%",
                         help="Percentage improvement in transmission success")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scenarios = ['Without Repeaters', 'With Repeaters']
            success_rates = [performance['without_repeaters'], performance['with_repeaters']]
            colors = ['lightcoral', 'lightgreen']
            
            bars = ax.bar(scenarios, success_rates, color=colors, alpha=0.7)
            ax.set_ylabel('Success Rate')
            ax.set_title('Quantum Transmission Success Rate Comparison')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.2%}', ha='center', va='bottom')
            
            # Add improvement annotation
            if improvement > 0:
                ax.annotate(f'+{improvement:.1f}% improvement', 
                           xy=(1, performance['with_repeaters']), 
                           xytext=(1.2, performance['with_repeaters'] + 0.1),
                           arrowprops=dict(arrowstyle='->', color='green'),
                           color='green', fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
            
            # Repeater details
            stats = repeater_system.get_repeater_stats()
            st.markdown(f"""
            **Repeater Configuration:**
            - **Repeaters placed:** {len(repeater_system.repeater_nodes)} at nodes {list(repeater_system.repeater_nodes)}
            - **Repeater efficiency:** {repeater_efficiency:.1%}
            - **Tests conducted:** {performance['tests_conducted']}
            - **Strategy:** Betweenness centrality (optimal positioning)
            """)
        
        # Step 6: Complete
        progress_bar.progress(100)
        status_text.text("Simulation complete!")
        
        st.success("✅ Simulation completed successfully!")
        
        # Download results option
        st.subheader("Export Results")
        
        if st.button("📊 Download Network Stats as CSV"):
            stats_df = pd.DataFrame([stats])
            csv = stats_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="network_stats.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        status_text.text("Simulation failed!")


def display_theory():
    """Display theoretical background."""
    st.header("📚 Theoretical Background")
    
    st.subheader("Quantum Networking Challenges")
    st.markdown("""
    - **Decoherence**: Quantum states degrade over distance and time
    - **No-Cloning Theorem**: Quantum information cannot be perfectly copied
    - **Entanglement Distribution**: Creating shared quantum states between distant nodes
    - **Quantum Error Correction**: Protecting quantum information from noise
    """)
    
    st.subheader("Hybrid Network Approach")
    st.markdown("""
    - **Quantum Nodes**: Capable of quantum processing and storage
    - **Classical Nodes**: Traditional networking equipment
    - **Hybrid Routing**: Intelligent path selection based on application needs
    - **Fallback Mechanisms**: Classical backup when quantum links fail
    """)

# Execute in tabs
with tab1:
    main()

with tab2:
    display_theory()
