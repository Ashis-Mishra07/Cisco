"""
Quantum link simulation module for modeling quantum networking challenges.
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List
import os
from network import HybridNetworkSimulator


class QuantumLinkSimulator:
    """Simulates quantum link behavior with realistic physics constraints."""
    
    def __init__(self, network: HybridNetworkSimulator, 
                 decoherence_rate: float = 0.01,
                 entanglement_success_base: float = 0.8,
                 classical_packet_loss_rate: float = 0.001,
                 no_cloning_violation_rate: float = 0.05):
        """Initialize quantum link simulator."""
        self.network = network
        self.decoherence_rate = decoherence_rate  # per km
        self.entanglement_success_base = entanglement_success_base
        self.classical_packet_loss_rate = classical_packet_loss_rate  # per km
        self.no_cloning_violation_rate = no_cloning_violation_rate
        
        # Latency parameters
        self.speed_of_light = 299792458  # m/s
        self.fiber_refractive_index = 1.46  # typical optical fiber
        self.quantum_processing_delay = 0.001  # 1ms for quantum operations
        self.classical_processing_delay = 0.0001  # 0.1ms for classical processing
        
    def calculate_latency(self, distance_km: float, link_type: str) -> float:
        """Calculate transmission latency based on distance and link type."""
        # Convert distance to meters
        distance_m = distance_km * 1000
        
        # Propagation delay through fiber
        effective_speed = self.speed_of_light / self.fiber_refractive_index
        propagation_delay = distance_m / effective_speed  # seconds
        
        # Processing delay based on link type
        if link_type == 'quantum':
            processing_delay = self.quantum_processing_delay
            # Additional quantum state preparation/measurement delay
            quantum_overhead = distance_km * 0.00001  # 10µs per km for quantum operations
            total_latency = propagation_delay + processing_delay + quantum_overhead
        else:
            processing_delay = self.classical_processing_delay
            total_latency = propagation_delay + processing_delay
            
        return total_latency * 1000  # Convert to milliseconds
        
    def simulate_quantum_transmission(self, node1: int, node2: int) -> Dict:
        """Simulate quantum link behavior with decoherence."""
        if (node1, node2) not in self.network.link_properties:
            edge = (node2, node1)
        else:
            edge = (node1, node2)
            
        link_props = self.network.link_properties[edge]
        distance = link_props['distance']
        
        # Calculate latency for this transmission
        latency_ms = self.calculate_latency(distance, link_props['type'])
        
        result = {
            'link_type': link_props['type'],
            'distance': distance,
            'latency_ms': latency_ms,
            'success': False,
            'reason': None,
            # Detailed breakdown of effects
            'decoherence_prob': 0,
            'entanglement_success_prob': 0,
            'packet_loss_prob': 0,
            'no_cloning_violation': False
        }
        
        if link_props['type'] == 'quantum':
            # Calculate decoherence probability
            decoherence_prob = 1 - np.exp(-self.decoherence_rate * distance)
            result['decoherence_prob'] = decoherence_prob
            
            if random.random() < decoherence_prob:
                result['reason'] = 'decoherence'
                return result
            
            # No-cloning check (simulated)
            if random.random() < self.no_cloning_violation_rate:
                result['reason'] = 'no-cloning violation'
                result['no_cloning_violation'] = True
                return result
            
            # Calculate entanglement distribution probability
            entanglement_success = self.entanglement_success_base * np.exp(-distance/1000)
            result['entanglement_success_prob'] = entanglement_success
            
            if random.random() < entanglement_success:
                result['success'] = True
                result['reason'] = 'successful quantum transmission'
            else:
                result['reason'] = 'entanglement failure'
        else:
            # Classical link - calculate packet loss probability
            packet_loss_prob = 1 - np.exp(-self.classical_packet_loss_rate * distance)
            result['packet_loss_prob'] = packet_loss_prob
            
            if random.random() > packet_loss_prob:
                result['success'] = True
                result['reason'] = 'successful classical transmission'
            else:
                result['reason'] = 'packet loss'
        
        return result
    
    def simulate_entanglement_swapping(self, path: List[int]) -> bool:
        """Simulate entanglement swapping along a path."""
        if len(path) < 2:
            return False
        
        # Check if all nodes in path are quantum
        for node in path:
            if self.network.G.nodes[node]['type'] != 'quantum':
                return False
        
        # Simulate swapping with decreasing probability
        success_prob = 0.9
        for i in range(len(path) - 1):
            if random.random() > success_prob:
                return False
            success_prob *= 0.95  # Degradation with each swap
        
        return True
    
    def analyze_link_behavior(self, num_simulations: int = 1000, save_path: str = None):
        """Analyze and plot link behavior differences including detailed physics effects."""
        results = {'quantum': [], 'classical': []}
        distances = {'quantum': [], 'classical': []}
        latencies = {'quantum': [], 'classical': []}
        
        # Track individual physics effects
        decoherence_failures = []
        entanglement_failures = []
        packet_loss_failures = []
        no_cloning_violations = []
        
        # Track effect probabilities for analysis
        decoherence_probs = []
        entanglement_probs = []
        packet_loss_probs = []
        
        for _ in range(num_simulations):
            edge = random.choice(list(self.network.G.edges()))
            result = self.simulate_quantum_transmission(edge[0], edge[1])
            
            link_type = result['link_type']
            results[link_type].append(1 if result['success'] else 0)
            distances[link_type].append(result['distance'])
            latencies[link_type].append(result['latency_ms'])
            
            # Track specific failure causes and probabilities
            if link_type == 'quantum':
                decoherence_probs.append(result['decoherence_prob'])
                entanglement_probs.append(result['entanglement_success_prob'])
                
                if result['reason'] == 'decoherence':
                    decoherence_failures.append(1)
                else:
                    decoherence_failures.append(0)
                    
                if result['reason'] == 'entanglement failure':
                    entanglement_failures.append(1)
                else:
                    entanglement_failures.append(0)
                    
                if result['no_cloning_violation']:
                    no_cloning_violations.append(1)
                else:
                    no_cloning_violations.append(0)
            else:
                packet_loss_probs.append(result['packet_loss_prob'])
                if result['reason'] == 'packet loss':
                    packet_loss_failures.append(1)
                else:
                    packet_loss_failures.append(0)
        
        # Create visualization with latency subplot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Success rates
        quantum_success = np.mean(results['quantum']) if results['quantum'] else 0
        classical_success = np.mean(results['classical']) if results['classical'] else 0
        
        bars = ax1.bar(['Quantum', 'Classical'], [quantum_success, classical_success])
        bars[0].set_color('red')
        bars[1].set_color('blue')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Link Success Rates')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, [quantum_success, classical_success]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.2%}', ha='center', va='bottom')
        
        # Success vs Distance
        if results['quantum']:
            ax2.scatter(distances['quantum'], results['quantum'], 
                       alpha=0.5, label='Quantum', color='red')
        if results['classical']:
            ax2.scatter(distances['classical'], results['classical'], 
                       alpha=0.5, label='Classical', color='blue')
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Success (1) / Failure (0)')
        ax2.set_title('Success vs Distance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Latency vs Distance
        if latencies['quantum']:
            ax3.scatter(distances['quantum'], latencies['quantum'], 
                       alpha=0.5, label='Quantum', color='red')
        if latencies['classical']:
            ax3.scatter(distances['classical'], latencies['classical'], 
                       alpha=0.5, label='Classical', color='blue')
        ax3.set_xlabel('Distance (km)')
        ax3.set_ylabel('Latency (ms)')
        ax3.set_title('Latency vs Distance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Link behavior analysis saved to {save_path}")
        
        plt.show()
        
        # Calculate and return detailed statistics including individual effects
        stats = {
            'quantum_success_rate': quantum_success,
            'classical_success_rate': classical_success,
            'quantum_avg_latency': np.mean(latencies['quantum']) if latencies['quantum'] else 0,
            'classical_avg_latency': np.mean(latencies['classical']) if latencies['classical'] else 0,
            'quantum_max_latency': np.max(latencies['quantum']) if latencies['quantum'] else 0,
            'classical_max_latency': np.max(latencies['classical']) if latencies['classical'] else 0,
            'total_tests': num_simulations,
            
            # Detailed physics effects breakdown
            'decoherence_failure_rate': np.mean(decoherence_failures) if decoherence_failures else 0,
            'entanglement_failure_rate': np.mean(entanglement_failures) if entanglement_failures else 0,
            'packet_loss_failure_rate': np.mean(packet_loss_failures) if packet_loss_failures else 0,
            'no_cloning_violation_rate': np.mean(no_cloning_violations) if no_cloning_violations else 0,
            
            # Average effect probabilities
            'avg_decoherence_prob': np.mean(decoherence_probs) if decoherence_probs else 0,
            'avg_entanglement_success_prob': np.mean(entanglement_probs) if entanglement_probs else 0,
            'avg_packet_loss_prob': np.mean(packet_loss_probs) if packet_loss_probs else 0,
            
            # Distance dependencies
            'quantum_avg_distance': np.mean(distances['quantum']) if distances['quantum'] else 0,
            'classical_avg_distance': np.mean(distances['classical']) if distances['classical'] else 0
        }
        
        return stats
