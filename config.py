"""
Configuration management for quantum network simulation.
"""
import yaml
import json
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class NetworkConfig:
    """Network configuration parameters."""
    num_nodes: int = 12
    quantum_node_ratio: float = 0.4
    prob_edge: float = 0.3
    distance_range: tuple = (10, 100)


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""
    decoherence_rate: float = 0.01
    entanglement_success_base: float = 0.8
    classical_packet_loss_rate: float = 0.001
    no_cloning_violation_rate: float = 0.05
    reliability_trials: int = 20
    
    # Latency parameters
    fiber_refractive_index: float = 1.46
    quantum_processing_delay: float = 0.001  # 1ms
    classical_processing_delay: float = 0.0001  # 0.1ms


@dataclass
class AnalysisConfig:
    """Analysis configuration parameters."""
    node_counts: list = None
    trials_per_size: int = 5
    num_simulations: int = 500
    num_repeater_tests: int = 100
    num_users_pki: int = 25


@dataclass
class OutputConfig:
    """Output configuration parameters."""
    output_dir: str = "./outputs"
    save_plots: bool = True
    save_csv: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration manager."""
        if config_file:
            self.load_config(config_file)
        else:
            self.load_default_config()
    
    def load_default_config(self):
        """Load default configuration."""
        self.network = NetworkConfig()
        self.simulation = SimulationConfig()
        self.analysis = AnalysisConfig(node_counts=[10, 20, 30, 50])
        self.output = OutputConfig()
    
    def load_config(self, config_file: str):
        """Load configuration from file."""
        try:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif config_file.endswith('.json'):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file}")
            
            # Parse configuration sections
            self.network = NetworkConfig(**config_data.get('network', {}))
            self.simulation = SimulationConfig(**config_data.get('simulation', {}))
            
            analysis_data = config_data.get('analysis', {})
            if 'node_counts' not in analysis_data:
                analysis_data['node_counts'] = [10, 20, 30, 50]
            self.analysis = AnalysisConfig(**analysis_data)
            
            self.output = OutputConfig(**config_data.get('output', {}))
            
            print(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            print(f"Error loading config file {config_file}: {e}")
            print("Using default configuration")
            self.load_default_config()
    
    def save_config(self, config_file: str):
        """Save current configuration to file."""
        config_data = {
            'network': self.network.__dict__,
            'simulation': self.simulation.__dict__,
            'analysis': self.analysis.__dict__,
            'output': self.output.__dict__
        }
        
        try:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                with open(config_file, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
            elif config_file.endswith('.json'):
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_file}")
            
            print(f"Configuration saved to {config_file}")
            
        except Exception as e:
            print(f"Error saving config file {config_file}: {e}")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return {
            'network': self.network.__dict__,
            'simulation': self.simulation.__dict__,
            'analysis': self.analysis.__dict__,
            'output': self.output.__dict__
        }
    
    def print_config(self):
        """Print current configuration."""
        print("="*50)
        print("CURRENT CONFIGURATION")
        print("="*50)
        
        print("\nNetwork Configuration:")
        for key, value in self.network.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nSimulation Configuration:")
        for key, value in self.simulation.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nAnalysis Configuration:")
        for key, value in self.analysis.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nOutput Configuration:")
        for key, value in self.output.__dict__.items():
            print(f"  {key}: {value}")
        
        print("="*50)


def create_default_config_file(filename: str = "config.yaml"):
    """Create a default configuration file."""
    config_manager = ConfigManager()
    config_manager.save_config(filename)
    print(f"Default configuration file created: {filename}")


if __name__ == "__main__":
    # Create default config file
    create_default_config_file()
