"""
Quantum repeater implementation for enhanced network performance.
"""
import matplotlib.pyplot as plt
import networkx as nx
import random
from typing import List, Dict
import os
from network import HybridNetworkSimulator
from link_sim import QuantumLinkSimulator
from routing import HybridRoutingProtocol


class QuantumRepeater:
    """Implements quantum repeaters for enhanced long-distance quantum communication."""
    
    def __init__(self, network: HybridNetworkSimulator, repeater_efficiency: float = 0.85):
        """Initialize quantum repeater system."""
        self.network = network
        self.repeater_nodes = set()
        self.repeater_efficiency = repeater_efficiency
        
    def place_repeaters(self, num_repeaters: int, strategy: str = 'centrality'):
        """Strategically place quantum repeaters in the network."""
        if strategy == 'centrality':
            self._place_by_centrality(num_repeaters)
        elif strategy == 'random':
            self._place_randomly(num_repeaters)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        print(f"Placed {len(self.repeater_nodes)} quantum repeaters at nodes: {self.repeater_nodes}")
        
    def _place_by_centrality(self, num_repeaters: int):
        """Place repeaters based on betweenness centrality."""
        # Use betweenness centrality to find optimal positions
        centrality = nx.betweenness_centrality(self.network.G)
        
        # Select top nodes for repeater placement (only quantum nodes)
        quantum_centrality = {node: centrality[node] for node in self.network.quantum_nodes}
        sorted_nodes = sorted(quantum_centrality.items(), key=lambda x: x[1], reverse=True)
        
        placed = 0
        for node, _ in sorted_nodes:
            if placed >= num_repeaters:
                break
            self.repeater_nodes.add(node)
            self.network.G.nodes[node]['has_repeater'] = True
            placed += 1
            
    def _place_randomly(self, num_repeaters: int):
        """Place repeaters randomly among quantum nodes."""
        available_nodes = list(self.network.quantum_nodes)
        num_to_place = min(num_repeaters, len(available_nodes))
        
        selected_nodes = random.sample(available_nodes, num_to_place)
        for node in selected_nodes:
            self.repeater_nodes.add(node)
            self.network.G.nodes[node]['has_repeater'] = True
        
    def enhanced_quantum_transmission(self, path: List[int]) -> bool:
        """Simulate quantum transmission with repeater support."""
        if len(path) < 2:
            return False
            
        success_prob = 0.9
        
        for i in range(len(path) - 1):
            # Check if current node has repeater
            if path[i] in self.repeater_nodes:
                success_prob = min(success_prob * self.repeater_efficiency, 0.9)
            else:
                success_prob *= 0.7  # Normal degradation
            
            if random.random() > success_prob:
                return False
                
        return True
    
    def compare_performance(self, num_tests: int = 100, save_path: str = None) -> Dict:
        """Compare network performance with and without repeaters."""
        link_sim = QuantumLinkSimulator(self.network)
        router = HybridRoutingProtocol(self.network, link_sim)
        
        results_without = {'success': 0, 'total': 0}
        results_with = {'success': 0, 'total': 0}
        
        # Ensure we have quantum nodes to test
        if len(self.network.quantum_nodes) < 2:
            print("Not enough quantum nodes for testing")
            return {'improvement': 0, 'without_repeaters': 0, 'with_repeaters': 0}
        
        for _ in range(num_tests):
            # Random source and destination from quantum nodes
            quantum_nodes_list = list(self.network.quantum_nodes)
            source = random.choice(quantum_nodes_list)
            dest = random.choice(quantum_nodes_list)
            
            if source != dest:
                path = router.find_best_path(source, dest, prefer_quantum=True)
                
                if path and len(path) > 1:
                    # Without repeaters
                    if link_sim.simulate_entanglement_swapping(path):
                        results_without['success'] += 1
                    results_without['total'] += 1
                    
                    # With repeaters
                    if self.enhanced_quantum_transmission(path):
                        results_with['success'] += 1
                    results_with['total'] += 1
        
        # Calculate success rates
        success_rate_without = (results_without['success'] / results_without['total'] 
                               if results_without['total'] > 0 else 0)
        success_rate_with = (results_with['success'] / results_with['total'] 
                            if results_with['total'] > 0 else 0)
        
        # Calculate improvement
        improvement = (((success_rate_with - success_rate_without) / success_rate_without) * 100 
                      if success_rate_without > 0 else 0)
        
        # Visualize results
        self._plot_comparison(success_rate_without, success_rate_with, improvement, save_path)
        
        print(f"Performance improvement with repeaters: {improvement:.1f}%")
        
        return {
            'improvement': improvement,
            'without_repeaters': success_rate_without,
            'with_repeaters': success_rate_with,
            'tests_conducted': results_without['total']
        }
    
    def _plot_comparison(self, rate_without: float, rate_with: float, 
                        improvement: float, save_path: str = None):
        """Plot performance comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        success_rates = [rate_without, rate_with]
        bars = ax.bar(['Without Repeaters', 'With Repeaters'], success_rates)
        bars[0].set_color('red')
        bars[1].set_color('green')
        
        ax.set_ylabel('Success Rate')
        ax.set_title(f'Quantum Transmission Success Rate\n(Improvement: {improvement:.1f}%)')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.2%}', ha='center', va='bottom')
        
        # Add improvement annotation
        if improvement != 0:
            ax.annotate(f'+{improvement:.1f}%', 
                       xy=(1, rate_with), xytext=(1.2, rate_with),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=12, color='green', weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Repeater performance comparison saved to {save_path}")
        
        plt.show()
    
    def get_repeater_stats(self) -> Dict:
        """Get statistics about placed repeaters."""
        if not self.repeater_nodes:
            return {'total_repeaters': 0}
            
        # Calculate coverage metrics
        total_quantum_nodes = len(self.network.quantum_nodes)
        coverage_ratio = len(self.repeater_nodes) / total_quantum_nodes if total_quantum_nodes > 0 else 0
        
        # Calculate network impact
        centrality = nx.betweenness_centrality(self.network.G)
        avg_repeater_centrality = (sum(centrality[node] for node in self.repeater_nodes) / 
                                  len(self.repeater_nodes))
        
        return {
            'total_repeaters': len(self.repeater_nodes),
            'coverage_ratio': coverage_ratio,
            'avg_centrality': avg_repeater_centrality,
            'efficiency': self.repeater_efficiency,
            'quantum_nodes_total': total_quantum_nodes
        }
