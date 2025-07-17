# Quantum-Classical Hybrid Network Simulation

A comprehensive simulation and analysis tool for quantum-classical hybrid networks, implementing all six parts of the Cisco quantum networking challenge.

## Features

- **Hybrid Network Topology**: Simulates mixed quantum-classical node networks
- **Quantum Physics Modeling**: Includes decoherence, no-cloning theorem, and entanglement distribution
- **Hybrid Routing Protocol**: Intelligent routing with quantum preference and classical fallback
- **Scalability Analysis**: Performance analysis across different network sizes
- **Quantum Repeaters**: Implementation and performance comparison
- **Symmetric Key PKI**: Comparison of three key distribution approaches
- **Configurable Parameters**: YAML/JSON configuration support
- **Command-Line Interface**: Easy-to-use CLI with extensive options
- **Export Capabilities**: Automatic plot and CSV generation

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone or download the project files**
   ```bash
   cd /path/to/quantum-network-simulation
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Usage
Run the simulation with default parameters:
```bash
python main.py
```

### Custom Parameters
```bash
# Run with 30 nodes and 1000 trials
python main.py --nodes 30 --trials 1000

# Use custom configuration
python main.py --config my_config.yaml

# Save to custom output directory
python main.py --output-dir ./my_results
```

### Configuration File
Create a custom configuration file:
```bash
python config.py  # Creates default config.yaml
```

Example configuration (`config.yaml`):
```yaml
network:
  num_nodes: 25
  quantum_node_ratio: 0.4
  prob_edge: 0.3
  distance_range: [10, 100]

simulation:
  decoherence_rate: 0.01
  entanglement_success_base: 0.8
  classical_packet_loss_rate: 0.001

analysis:
  node_counts: [10, 20, 30, 50, 100]
  trials_per_size: 10
  num_simulations: 1000

output:
  output_dir: "./outputs"
  save_plots: true
  save_csv: true
  plot_format: "png"
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--nodes` | Number of nodes in network | 12 |
| `--quantum-ratio` | Ratio of quantum nodes (0.0-1.0) | 0.4 |
| `--trials` | Number of simulation trials | 500 |
| `--scalability-trials` | Trials per network size | 5 |
| `--config` | Configuration file path | None |
| `--output-dir` | Output directory | ./outputs |
| `--no-plots` | Disable plot generation | False |
| `--no-csv` | Disable CSV export | False |

## Expected Output

After running the simulation, check the output directory for:

### Generated Files
- `network_topology.png` - Network visualization
- `link_behavior.png` - Quantum vs classical link analysis
- `scalability_analysis.png` - Performance vs network size
- `repeater_performance.png` - Repeater effectiveness comparison
- `pki_comparison.png` - Key distribution method comparison
- `scalability_results.csv` - Detailed scalability data
- `pki_comparison.csv` - PKI comparison table
- `config_used.yaml` - Configuration used for the run

### Console Output
```
QUANTUM-CLASSICAL HYBRID NETWORK SIMULATION
============================================================

Part 1: Creating Hybrid Network...
Part 2: Simulating Quantum Networking Challenges...
Quantum Success Rate: 66.67%
Classical Success Rate: 95.32%

Part 3: Testing Hybrid Routing Protocol...
✓ Message delivered successfully via quantum path
Path: 0 -> 3 -> 9 -> 8

Part 4: Analyzing Scalability...
[Scalability analysis results...]

Part 5: Implementing Quantum Repeaters...
Performance improvement with repeaters: 92.3%

Part 6: Designing Symmetric Key PKI System...
[PKI comparison results...]

SIMULATION COMPLETE
============================================================
```

## Module Structure

- `main.py` - Main entry point and CLI
- `config.py` - Configuration management
- `network.py` - Network topology simulation
- `link_sim.py` - Quantum link behavior modeling
- `routing.py` - Hybrid routing protocol
- `analysis.py` - Scalability and performance analysis
- `repeater.py` - Quantum repeater implementation
- `pki.py` - Symmetric key distribution systems

## Research Applications

This simulation addresses key challenges in quantum networking:

1. **Decoherence Effects**: Models realistic quantum state degradation
2. **No-Cloning Theorem**: Simulates fundamental quantum limitations
3. **Hybrid Network Design**: Explores quantum-classical integration
4. **Scalability Analysis**: Identifies bottlenecks in large networks
5. **Infrastructure Requirements**: Evaluates repeater effectiveness
6. **Security Protocols**: Compares key distribution methods

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows style guidelines
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact the development team.
