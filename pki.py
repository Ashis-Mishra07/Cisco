"""
PKI (Public Key Infrastructure) without public keys - symmetric key distribution systems.
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from typing import Dict, List, Tuple
import os


class SymmetricKeyPKI:
    """Implements and compares symmetric key distribution approaches."""
    
    def __init__(self, num_users: int = 25):
        """Initialize symmetric key PKI system."""
        self.num_users = num_users
        self.users = list(range(num_users))
        
    def option1_pairwise_keys(self) -> Dict:
        """Option 1: Each pair shares a unique symmetric key."""
        total_keys = (self.num_users * (self.num_users - 1)) // 2
        
        print(f"\nOption 1: Pairwise Keys")
        print(f"Total keys needed: {total_keys}")
        print(f"Keys per user: {self.num_users - 1}")
        
        # Simulate key distribution
        keys = {}
        for i in range(self.num_users):
            for j in range(i + 1, self.num_users):
                keys[(i, j)] = f"key_{i}_{j}"
        
        return {
            'method': 'pairwise',
            'total_keys': total_keys,
            'keys_per_user': self.num_users - 1,
            'scalability': 'O(n²)',
            'pros': ['Perfect forward secrecy', 'No single point of failure'],
            'cons': ['Poor scalability', 'High key management overhead'],
            'keys': keys
        }
    
    def option2_kdc(self) -> Dict:
        """Option 2: Key Distribution Center (KDC) approach."""
        print(f"\nOption 2: Key Distribution Center")
        print(f"Total keys needed: {self.num_users}")
        print(f"Keys per user: 1")
        
        return {
            'method': 'KDC',
            'total_keys': self.num_users,
            'keys_per_user': 1,
            'scalability': 'O(n)',
            'pros': ['Excellent scalability', 'Simple key management'],
            'cons': ['Single point of failure', 'KDC must be trusted']
        }
    
    def option3_group_key_hierarchy(self) -> Dict:
        """Option 3: Hierarchical group key management."""
        group_size = int(math.sqrt(self.num_users))
        num_groups = math.ceil(self.num_users / group_size)
        
        print(f"\nOption 3: Hierarchical Group Keys")
        print(f"Group size: {group_size}")
        print(f"Number of groups: {num_groups}")
        
        # Calculate total keys: group keys + intra-group pairwise keys
        intra_group_keys = (group_size * (group_size - 1) // 2) * num_groups
        total_keys = num_groups + intra_group_keys
        
        return {
            'method': 'hierarchical',
            'total_keys': total_keys,
            'keys_per_user': group_size,
            'scalability': 'O(√n)',
            'pros': ['Balance between security and scalability', 'Partial compromise containment'],
            'cons': ['More complex than KDC', 'Inter-group communication overhead'],
            'group_size': group_size,
            'num_groups': num_groups
        }
    
    def implement_kdc_system(self):
        """Implement a simple KDC-based system."""
        class KDC:
            def __init__(self, users):
                self.users = users
                self.master_keys = {user: f"master_key_{user}" for user in users}
                self.session_keys = {}
                
            def request_session_key(self, user1, user2):
                """Generate session key for communication between two users."""
                if user1 not in self.users or user2 not in self.users:
                    return None
                    
                session_id = f"{min(user1, user2)}_{max(user1, user2)}_{random.randint(1000, 9999)}"
                session_key = f"session_{session_id}"
                
                # In real implementation, these would be encrypted with master keys
                ticket1 = {
                    'session_key': session_key,
                    'peer': user2,
                    'timestamp': 'current_time'
                }
                ticket2 = {
                    'session_key': session_key,
                    'peer': user1,
                    'timestamp': 'current_time'
                }
                
                return ticket1, ticket2
        
        # Example usage
        kdc = KDC(self.users)
        
        # Simulate communication between users 3 and 7
        user1, user2 = 3, 7
        ticket1, ticket2 = kdc.request_session_key(user1, user2)
        
        print(f"\nKDC Implementation Example:")
        print(f"User {user1} receives: {ticket1}")
        print(f"User {user2} receives: {ticket2}")
        print(f"Both users can now communicate using session key: {ticket1['session_key']}")
        
        return kdc
    
    def compare_approaches(self, save_path: str = None) -> Dict:
        """Compare all three approaches."""
        options = [
            self.option1_pairwise_keys(),
            self.option2_kdc(),
            self.option3_group_key_hierarchy()
        ]
        
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Total keys comparison
        methods = [opt['method'] for opt in options]
        total_keys = [opt['total_keys'] for opt in options]
        
        bars1 = ax1.bar(methods, total_keys)
        ax1.set_ylabel('Total Keys Required')
        ax1.set_title('Key Storage Requirements')
        
        # Color code by efficiency
        colors1 = ['red', 'green', 'orange']
        for bar, color in zip(bars1, colors1):
            bar.set_color(color)
        
        # Add value labels
        for bar, keys in zip(bars1, total_keys):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{keys}', ha='center', va='bottom')
        
        # Scalability comparison for different network sizes
        network_sizes = [10, 25, 50, 100, 200]
        
        pairwise_keys = [(n * (n - 1)) // 2 for n in network_sizes]
        kdc_keys = network_sizes
        hierarchical_keys = [int(math.sqrt(n)) * n for n in network_sizes]
        
        ax2.plot(network_sizes, pairwise_keys, 'r-o', label='Pairwise', linewidth=2)
        ax2.plot(network_sizes, kdc_keys, 'g-o', label='KDC', linewidth=2)
        ax2.plot(network_sizes, hierarchical_keys, 'orange', marker='o', label='Hierarchical', linewidth=2)
        ax2.set_xlabel('Number of Users')
        ax2.set_ylabel('Total Keys Required')
        ax2.set_title('Scalability Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PKI comparison saved to {save_path}")
        
        plt.show()
        
        # Print detailed comparison
        print("\n" + "="*60)
        print("SYMMETRIC KEY DISTRIBUTION COMPARISON")
        print("="*60)
        
        comparison_results = {}
        for opt in options:
            print(f"\n{opt['method'].upper()} Approach:")
            print(f"  - Total keys: {opt['total_keys']}")
            print(f"  - Keys per user: {opt['keys_per_user']}")
            print(f"  - Scalability: {opt['scalability']}")
            print(f"  - Pros: {', '.join(opt['pros'])}")
            print(f"  - Cons: {', '.join(opt['cons'])}")
            
            comparison_results[opt['method']] = {
                'total_keys': opt['total_keys'],
                'keys_per_user': opt['keys_per_user'],
                'scalability': opt['scalability']
            }
        
        return comparison_results
    
    def export_comparison_table(self, filepath: str):
        """Export comparison table to CSV."""
        options = [
            self.option1_pairwise_keys(),
            self.option2_kdc(),
            self.option3_group_key_hierarchy()
        ]
        
        # Create comparison data
        comparison_data = []
        for opt in options:
            comparison_data.append({
                'Method': opt['method'],
                'Total Keys': opt['total_keys'],
                'Keys per User': opt['keys_per_user'],
                'Scalability': opt['scalability'],
                'Pros': '; '.join(opt['pros']),
                'Cons': '; '.join(opt['cons'])
            })
        
        # Write to CSV (manual implementation to avoid pandas dependency)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            # Write header
            headers = comparison_data[0].keys()
            f.write(','.join(headers) + '\n')
            
            # Write data
            for row in comparison_data:
                values = [str(row[header]).replace(',', ';') for header in headers]
                f.write(','.join(values) + '\n')
        
        print(f"PKI comparison table exported to {filepath}")
    
    @staticmethod
    def calculate_key_requirements(num_users: int) -> Dict[str, int]:
        """Calculate key requirements for different approaches for given number of users."""
        return {
            'pairwise': (num_users * (num_users - 1)) // 2,
            'kdc': num_users,
            'hierarchical': int(math.sqrt(num_users)) * num_users
        }
