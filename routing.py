"""
Hybrid routing protocol for quantum-classical networks.
"""
from typing import Dict, List, Optional
import random
from network import HybridNetworkSimulator
from link_sim import QuantumLinkSimulator


class HybridRoutingProtocol:
    """Implements hybrid routing with quantum preference and classical fallback."""
    
    def __init__(self, network: HybridNetworkSimulator, 
                 link_simulator: QuantumLinkSimulator,
                 reliability_trials: int = 20):
        """Initialize hybrid routing protocol."""
        self.network = network
        self.link_sim = link_simulator
        self.reliability_trials = reliability_trials
        self.routing_table = {}
        self._build_routing_table()
    
    def _build_routing_table(self):
        """Build routing table with link reliability scores."""
        for edge in self.network.G.edges():
            # Simulate link multiple times to estimate reliability
            successes = 0
            
            for _ in range(self.reliability_trials):
                result = self.link_sim.simulate_quantum_transmission(edge[0], edge[1])
                if result['success']:
                    successes += 1
            
            # Ensure minimum reliability to avoid division by zero
            reliability = max(successes / self.reliability_trials, 0.001)
            self.routing_table[edge] = {
                'reliability': reliability,
                'type': self.network.link_properties[edge]['type'],
                'distance': self.network.link_properties[edge]['distance']
            }
    
    def find_best_path(self, source: int, destination: int, 
                      prefer_quantum: bool = True) -> Optional[List[int]]:
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
                    edge = ((current, neighbor) if (current, neighbor) in self.routing_table 
                           else (neighbor, current))
                    
                    reliability = self.routing_table[edge]['reliability']
                    link_type = self.routing_table[edge]['type']
                    distance = self.routing_table[edge]['distance']
                    
                    # Weight calculation with safety check for zero reliability
                    if reliability == 0:
                        weight = float('inf')  # Avoid completely unreliable links
                    elif prefer_quantum and link_type == 'quantum':
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
            'message': message
        }
        
        # First try quantum path
        quantum_path = self.find_best_path(source, destination, prefer_quantum=True)
        
        if quantum_path:
            # Attempt quantum transmission
            quantum_success = True
            for i in range(len(quantum_path) - 1):
                trans_result = self.link_sim.simulate_quantum_transmission(
                    quantum_path[i], quantum_path[i+1]
                )
                result['attempts'].append({
                    'type': 'quantum',
                    'link': (quantum_path[i], quantum_path[i+1]),
                    'success': trans_result['success'],
                    'reason': trans_result['reason'],
                    'latency_ms': trans_result['latency_ms']
                })
                
                if not trans_result['success']:
                    quantum_success = False
                    break
            
            if quantum_success:
                result['success'] = True
                result['path'] = quantum_path
                result['final_method'] = 'quantum'
                return result
        
        # Fallback to classical path
        classical_path = self.find_best_path(source, destination, prefer_quantum=False)
        
        if classical_path:
            classical_success = True
            for i in range(len(classical_path) - 1):
                trans_result = self.link_sim.simulate_quantum_transmission(
                    classical_path[i], classical_path[i+1]
                )
                result['attempts'].append({
                    'type': 'classical',
                    'link': (classical_path[i], classical_path[i+1]),
                    'success': trans_result['success'],
                    'reason': trans_result['reason'],
                    'latency_ms': trans_result['latency_ms']
                })
                
                if not trans_result['success']:
                    classical_success = False
                    break
            
            if classical_success:
                result['success'] = True
                result['path'] = classical_path
                result['final_method'] = 'classical'
        
        return result
    
    def get_routing_stats(self) -> Dict:
        """Get routing table statistics."""
        if not self.routing_table:
            return {}
            
        reliabilities = [edge_data['reliability'] for edge_data in self.routing_table.values()]
        quantum_reliabilities = [edge_data['reliability'] for edge_data in self.routing_table.values() 
                               if edge_data['type'] == 'quantum']
        classical_reliabilities = [edge_data['reliability'] for edge_data in self.routing_table.values() 
                                 if edge_data['type'] == 'classical']
        
        return {
            'total_links': len(self.routing_table),
            'avg_reliability': sum(reliabilities) / len(reliabilities),
            'min_reliability': min(reliabilities),
            'max_reliability': max(reliabilities),
            'avg_quantum_reliability': (sum(quantum_reliabilities) / len(quantum_reliabilities) 
                                      if quantum_reliabilities else 0),
            'avg_classical_reliability': (sum(classical_reliabilities) / len(classical_reliabilities) 
                                        if classical_reliabilities else 0),
            'quantum_links': len(quantum_reliabilities),
            'classical_links': len(classical_reliabilities)
        }
