"""
Main entry point for the quantum network simulation.
Command-line interface and orchestration of all simulation components.
"""
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Import our modules
from config import ConfigManager
from network import HybridNetworkSimulator
from link_sim import QuantumLinkSimulator
from routing import HybridRoutingProtocol
from analysis import ScalabilityAnalyzer
from repeater import QuantumRepeater
from pki import SymmetricKeyPKI


class QuantumNetworkSimulation:
    """Main simulation orchestrator."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize simulation with configuration."""
        self.config = config_manager
        self.results = {}
        
        # Create output directory
        os.makedirs(self.config.output.output_dir, exist_ok=True)
        
    def run_complete_simulation(self) -> Dict[str, Any]:
        """Run the complete quantum-classical hybrid network simulation."""
        
        print("="*60)
        print("QUANTUM-CLASSICAL HYBRID NETWORK SIMULATION")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Part 1: Create and visualize network
        print("\nPart 1: Creating Hybrid Network...")
        network = self._create_network()
        
        # Part 2: Simulate quantum challenges
        print("\nPart 2: Simulating Quantum Networking Challenges...")
        link_behavior = self._analyze_link_behavior(network)
        
        # Part 3: Test hybrid routing
        print("\nPart 3: Testing Hybrid Routing Protocol...")
        routing_results = self._test_routing(network)
        
        # Part 4: Scalability analysis
        print("\nPart 4: Analyzing Scalability...")
        scalability_results = self._analyze_scalability()
        
        # Part 5: Quantum repeater implementation
        print("\nPart 5: Implementing Quantum Repeaters...")
        repeater_results = self._test_repeaters(network)
        
        # Part 6: Symmetric Key PKI
        print("\nPart 6: Designing Symmetric Key PKI System...")
        pki_results = self._analyze_pki()
        
        # Compile results
        self.results = {
            'network_stats': network.get_network_stats(),
            'link_behavior': link_behavior,
            'routing_results': routing_results,
            'scalability_results': scalability_results,
            'repeater_results': repeater_results,
            'pki_results': pki_results,
            'config': self.config.get_config_dict()
        }
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        print(f"Results saved to: {self.config.output.output_dir}")
        
        return self.results
    
    def _create_network(self) -> HybridNetworkSimulator:
        """Create and visualize the network."""
        network = HybridNetworkSimulator(
            num_nodes=self.config.network.num_nodes,
            quantum_node_ratio=self.config.network.quantum_node_ratio,
            prob_edge=self.config.network.prob_edge,
            distance_range=self.config.network.distance_range
        )
        
        if self.config.output.save_plots:
            save_path = os.path.join(self.config.output.output_dir, 
                                   f"network_topology.{self.config.output.plot_format}")
            network.visualize_network(save_path)
        else:
            network.visualize_network()
            
        return network
    
    def _analyze_link_behavior(self, network: HybridNetworkSimulator) -> Dict:
        """Analyze quantum vs classical link behavior."""
        link_sim = QuantumLinkSimulator(
            network,
            decoherence_rate=self.config.simulation.decoherence_rate,
            entanglement_success_base=self.config.simulation.entanglement_success_base,
            classical_packet_loss_rate=self.config.simulation.classical_packet_loss_rate,
            no_cloning_violation_rate=self.config.simulation.no_cloning_violation_rate
        )
        
        save_path = None
        if self.config.output.save_plots:
            save_path = os.path.join(self.config.output.output_dir, 
                                   f"link_behavior.{self.config.output.plot_format}")
        
        results = link_sim.analyze_link_behavior(
            num_simulations=self.config.analysis.num_simulations,
            save_path=save_path
        )
        
        print(f"Quantum Success Rate: {results['quantum_success_rate']:.2%}")
        print(f"Classical Success Rate: {results['classical_success_rate']:.2%}")
        print(f"Quantum Average Latency: {results['quantum_avg_latency']:.2f}ms")
        print(f"Classical Average Latency: {results['classical_avg_latency']:.2f}ms")
        print(f"Quantum Max Latency: {results['quantum_max_latency']:.2f}ms")
        print(f"Classical Max Latency: {results['classical_max_latency']:.2f}ms")
        
        # Display detailed physics effects breakdown
        print(f"\n--- Detailed Physics Effects ---")
        print(f"Decoherence Failure Rate: {results['decoherence_failure_rate']:.2%}")
        print(f"Entanglement Failure Rate: {results['entanglement_failure_rate']:.2%}")
        print(f"Classical Packet Loss Rate: {results['packet_loss_failure_rate']:.2%}")
        print(f"No-Cloning Violations: {results['no_cloning_violation_rate']:.2%}")
        print(f"\n--- Average Effect Probabilities ---")
        print(f"Average Decoherence Probability: {results['avg_decoherence_prob']:.2%}")
        print(f"Average Entanglement Success Probability: {results['avg_entanglement_success_prob']:.2%}")
        print(f"Average Packet Loss Probability: {results['avg_packet_loss_prob']:.2%}")
        
        return results
    
    def _test_routing(self, network: HybridNetworkSimulator) -> Dict:
        """Test hybrid routing protocol."""
        link_sim = QuantumLinkSimulator(
            network,
            decoherence_rate=self.config.simulation.decoherence_rate,
            entanglement_success_base=self.config.simulation.entanglement_success_base,
            classical_packet_loss_rate=self.config.simulation.classical_packet_loss_rate
        )
        
        router = HybridRoutingProtocol(
            network, 
            link_sim, 
            reliability_trials=self.config.simulation.reliability_trials
        )
        
        # Test message sending between random nodes
        source, dest = 0, min(8, network.num_nodes - 1)
        print(f"\nSending message from Node {source} to Node {dest}...")
        result = router.send_message(source, dest, "Hello Quantum World!")
        
        if result['success']:
            print(f"✓ Message delivered successfully via {result['final_method']} path")
            print(f"  Path: {' -> '.join(map(str, result['path']))}")
        else:
            print("✗ Message delivery failed")
        
        print(f"\nTransmission attempts:")
        total_latency = 0
        for attempt in result['attempts'][:5]:  # Show first 5 attempts
            status = "✓" if attempt['success'] else "✗"
            latency = attempt.get('latency_ms', 0)
            total_latency += latency
            print(f"  {status} {attempt['type']} link {attempt['link']}: {attempt['reason']} ({latency:.2f}ms)")
        
        if result['attempts']:
            avg_latency = total_latency / len(result['attempts'][:5])
            print(f"  Average latency for path: {avg_latency:.2f}ms")
        
        routing_stats = router.get_routing_stats()
        return {
            'message_result': result,
            'routing_stats': routing_stats
        }
    
    def _analyze_scalability(self) -> Dict:
        """Analyze network scalability."""
        analyzer = ScalabilityAnalyzer()
        
        results_summary = analyzer.analyze_scalability(
            node_counts=self.config.analysis.node_counts,
            trials_per_size=self.config.analysis.trials_per_size,
            quantum_node_ratio=self.config.network.quantum_node_ratio
        )
        
        # Save plots
        if self.config.output.save_plots:
            save_path = os.path.join(self.config.output.output_dir, 
                                   f"scalability_analysis.{self.config.output.plot_format}")
            analyzer.plot_scalability_results(save_path)
        else:
            analyzer.plot_scalability_results()
        
        # Save CSV
        if self.config.output.save_csv:
            csv_path = os.path.join(self.config.output.output_dir, "scalability_results.csv")
            analyzer.export_results_to_csv(csv_path)
        
        # Generate report
        report = analyzer.generate_report()
        print(report)
        
        return {
            'summary': results_summary,
            'report': report
        }
    
    def _test_repeaters(self, network: HybridNetworkSimulator) -> Dict:
        """Test quantum repeater performance."""
        repeater_system = QuantumRepeater(network)
        repeater_system.place_repeaters(num_repeaters=3)
        
        save_path = None
        if self.config.output.save_plots:
            save_path = os.path.join(self.config.output.output_dir, 
                                   f"repeater_performance.{self.config.output.plot_format}")
        
        performance = repeater_system.compare_performance(
            num_tests=self.config.analysis.num_repeater_tests,
            save_path=save_path
        )
        
        stats = repeater_system.get_repeater_stats()
        
        return {
            'performance': performance,
            'stats': stats
        }
    
    def _analyze_pki(self) -> Dict:
        """Analyze PKI approaches."""
        pki = SymmetricKeyPKI(num_users=self.config.analysis.num_users_pki)
        
        save_path = None
        if self.config.output.save_plots:
            save_path = os.path.join(self.config.output.output_dir, 
                                   f"pki_comparison.{self.config.output.plot_format}")
        
        comparison_results = pki.compare_approaches(save_path)
        
        # Save CSV
        if self.config.output.save_csv:
            csv_path = os.path.join(self.config.output.output_dir, "pki_comparison.csv")
            pki.export_comparison_table(csv_path)
        
        # Implement KDC system
        kdc = pki.implement_kdc_system()
        
        return {
            'comparison': comparison_results,
            'kdc_example': "KDC system implemented successfully"
        }


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Quantum-Classical Hybrid Network Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Network parameters
    parser.add_argument('--nodes', type=int, default=12,
                       help='Number of nodes in the network')
    parser.add_argument('--quantum-ratio', type=float, default=0.4,
                       help='Ratio of quantum nodes (0.0-1.0)')
    
    # Analysis parameters
    parser.add_argument('--trials', type=int, default=500,
                       help='Number of simulation trials')
    parser.add_argument('--scalability-trials', type=int, default=5,
                       help='Trials per network size for scalability analysis')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file (YAML or JSON)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for results')
    
    # Output options
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--no-csv', action='store_true',
                       help='Disable CSV export')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config_manager = ConfigManager(args.config)
    else:
        config_manager = ConfigManager()
    
    # Override config with command-line arguments
    if args.nodes != 12:
        config_manager.network.num_nodes = args.nodes
    if args.quantum_ratio != 0.4:
        config_manager.network.quantum_node_ratio = args.quantum_ratio
    if args.trials != 500:
        config_manager.analysis.num_simulations = args.trials
    if args.scalability_trials != 5:
        config_manager.analysis.trials_per_size = args.scalability_trials
    if args.output_dir != './outputs':
        config_manager.output.output_dir = args.output_dir
    if args.no_plots:
        config_manager.output.save_plots = False
    if args.no_csv:
        config_manager.output.save_csv = False
    
    # Print configuration
    config_manager.print_config()
    
    # Run simulation
    try:
        simulation = QuantumNetworkSimulation(config_manager)
        results = simulation.run_complete_simulation()
        
        # Save configuration used
        config_path = os.path.join(config_manager.output.output_dir, "config_used.yaml")
        config_manager.save_config(config_path)
        
        print(f"\nSimulation completed successfully!")
        print(f"Check {config_manager.output.output_dir} for results.")
        
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
