###############################################################################
# Political2025 - analyze party flip behaviors
#
# Inspired by Amy Shira Teitel.
#
# Author:      Phil Moyer (phil@moyer.ai)
# Date:        June 2025
# Copyright(c) 2025 Philip R. Moyer. All rights reserved.
#
###############################################################################

######################
# Import Libraries
######################

# Standard libraries modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
from collections import defaultdict
import pandas as pd

# Third-party modules

# Package/application modules


######################
# Globals
######################



######################
# Classes and Methods
######################

@dataclass
class PoliticalPlatform:
    """Represents a political party's platform as a vector of policy positions"""
    positions: np.ndarray  # Each dimension represents a policy issue (e.g., economic, social, etc.)
    party_id: int
    
    def distance_to(self, other: 'PoliticalPlatform') -> float:
        """Calculate ideological distance between platforms"""
        return np.linalg.norm(self.positions - other.positions)

@dataclass
class Voter:
    """Represents an individual voter with preferences"""
    preferences: np.ndarray  # Personal policy preferences
    party_affiliation: int   # Which party they currently support
    loyalty: float          # How resistant to changing parties (0-1)
    
class PoliticalEvolutionSimulator:
    def __init__(self, 
                 num_parties: int = 2,
                 num_voters: int = 10000,
                 num_dimensions: int = 8,  # Number of policy dimensions
                 mutation_rate: float = 0.02,
                 crossover_rate: float = 0.1):
        
        self.num_parties = num_parties
        self.num_voters = num_voters
        self.num_dimensions = num_dimensions
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initialize parties with distinct starting positions
        self.parties = self._initialize_parties()
        self.voters = self._initialize_voters()
        
        # History tracking
        self.party_history = []
        self.voter_distribution_history = []
        self.election_results_history = []
        
    def _initialize_parties(self) -> List[PoliticalPlatform]:
        """Initialize parties with historically accurate starting positions"""
        parties = []
        
        if self.num_parties == 2:
            # Historical Lincoln-era positions
            # Republicans: Anti-slavery, pro-business, federal power
            # Democrats: Pro-states rights, rural interests, limited federal power
            republican_start = np.array([0.7, -0.3, 0.6, 0.4, 0.5, -0.2, 0.3, 0.8])  # Pro-civil rights, pro-federal power
            democrat_start = np.array([-0.5, 0.6, -0.4, -0.6, -0.3, 0.4, -0.7, -0.2])  # Pro-states rights, rural
            
            parties.append(PoliticalPlatform(republican_start, 0))
            parties.append(PoliticalPlatform(democrat_start, 1))
        else:
            # For multiple parties, distribute them in ideological space
            for i in range(self.num_parties):
                angle = 2 * np.pi * i / self.num_parties
                base_position = np.array([np.cos(angle), np.sin(angle)] + 
                                       [random.uniform(-0.5, 0.5) for _ in range(self.num_dimensions - 2)])
                parties.append(PoliticalPlatform(base_position, i))
        
        return parties
    
    def _initialize_voters(self) -> List[Voter]:
        """Initialize voters with diverse preferences and loyalties"""
        voters = []
        
        for _ in range(self.num_voters):
            # Generate diverse voter preferences
            preferences = np.random.normal(0, 0.6, self.num_dimensions)
            
            # Assign initial party affiliation based on closest platform
            closest_party = self._find_closest_party(preferences)
            
            # Loyalty varies - some voters are very loyal, others swing easily
            loyalty = np.random.beta(2, 2)  # Beta distribution gives realistic loyalty spread
            
            voters.append(Voter(preferences, closest_party, loyalty))
        
        return voters
    
    def _find_closest_party(self, preferences: np.ndarray) -> int:
        """Find which party platform is closest to given preferences"""
        min_distance = float('inf')
        closest_party = 0
        
        for party in self.parties:
            distance = np.linalg.norm(preferences - party.positions)
            if distance < min_distance:
                min_distance = distance
                closest_party = party.party_id
        
        return closest_party
    
    def _calculate_fitness(self, party: PoliticalPlatform) -> float:
        """Calculate party fitness based on voter support and electoral success"""
        supporters = [v for v in self.voters if v.party_affiliation == party.party_id]
        base_support = len(supporters) / self.num_voters
        
        # Factor in how well the party represents its supporters
        if supporters:
            avg_distance = np.mean([
                np.linalg.norm(party.positions - voter.preferences) 
                for voter in supporters
            ])
            representation_quality = 1 / (1 + avg_distance)  # Closer = better
        else:
            representation_quality = 0
        
        return base_support * 0.7 + representation_quality * 0.3
    
    def _mutate_party(self, party: PoliticalPlatform) -> PoliticalPlatform:
        """Apply mutation to party platform"""
        new_positions = party.positions.copy()
        
        for i in range(len(new_positions)):
            if random.random() < self.mutation_rate:
                # Mutation with environmental pressure
                mutation_strength = random.gauss(0, 0.1)
                new_positions[i] += mutation_strength
                # Keep positions bounded
                new_positions[i] = np.clip(new_positions[i], -2, 2)
        
        return PoliticalPlatform(new_positions, party.party_id)
    
    def _crossover_parties(self, party1: PoliticalPlatform, party2: PoliticalPlatform) -> PoliticalPlatform:
        """Create hybrid platform through crossover"""
        if random.random() > self.crossover_rate:
            return party1
        
        # Weighted average based on relative fitness
        fitness1 = self._calculate_fitness(party1)
        fitness2 = self._calculate_fitness(party2)
        total_fitness = fitness1 + fitness2
        
        if total_fitness > 0:
            weight1 = fitness1 / total_fitness
            new_positions = weight1 * party1.positions + (1 - weight1) * party2.positions
        else:
            new_positions = 0.5 * (party1.positions + party2.positions)
        
        return PoliticalPlatform(new_positions, party1.party_id)
    
    def _update_voter_affiliations(self):
        """Update voter party affiliations based on current platforms"""
        for voter in self.voters:
            current_party = self.parties[voter.party_affiliation]
            current_distance = np.linalg.norm(voter.preferences - current_party.positions)
            
            # Check if another party is significantly better
            best_party_id = voter.party_affiliation
            best_distance = current_distance
            
            for party in self.parties:
                distance = np.linalg.norm(voter.preferences - party.positions)
                if distance < best_distance:
                    best_distance = distance
                    best_party_id = party.party_id
            
            # Switch parties based on loyalty and ideological distance
            if best_party_id != voter.party_affiliation:
                switching_probability = (1 - voter.loyalty) * (current_distance - best_distance)
                if random.random() < switching_probability:
                    voter.party_affiliation = best_party_id
    
    def _evolve_voter_preferences(self):
        """Slowly evolve voter preferences over time (generational change, major events)"""
        for voter in self.voters:
            # Small random drift in preferences
            if random.random() < 0.05:  # 5% chance per generation
                drift = np.random.normal(0, 0.05, self.num_dimensions)
                voter.preferences += drift
                voter.preferences = np.clip(voter.preferences, -2, 2)
    
    def simulate_generation(self):
        """Simulate one generation of political evolution"""
        # Update voter affiliations based on current party positions
        self._update_voter_affiliations()
        
        # Evolve party platforms
        new_parties = []
        for party in self.parties:
            # Mutate based on electoral pressure and internal dynamics
            mutated_party = self._mutate_party(party)
            
            # Potential crossover with other parties (coalition building, idea borrowing)
            if len(self.parties) > 1 and random.random() < self.crossover_rate:
                other_party = random.choice([p for p in self.parties if p.party_id != party.party_id])
                mutated_party = self._crossover_parties(mutated_party, other_party)
            
            new_parties.append(mutated_party)
        
        self.parties = new_parties
        
        # Gradually evolve voter preferences
        self._evolve_voter_preferences()
        
        # Record history
        self._record_generation()
    
    def _record_generation(self):
        """Record current state for analysis"""
        # Party positions
        party_positions = {f'party_{p.party_id}': p.positions.copy() for p in self.parties}
        self.party_history.append(party_positions)
        
        # Voter distribution
        distribution = defaultdict(int)
        for voter in self.voters:
            distribution[voter.party_affiliation] += 1
        self.voter_distribution_history.append(dict(distribution))
        
        # Election results (simplified)
        results = {party.party_id: self._calculate_fitness(party) for party in self.parties}
        self.election_results_history.append(results)
    
    def run_simulation(self, generations: int = 200):
        """Run the full simulation"""
        print(f"Running political evolution simulation with {self.num_parties} parties for {generations} generations...")
        
        # Record initial state
        self._record_generation()
        
        for generation in range(generations):
            if generation % 50 == 0:
                print(f"Generation {generation}/{generations}")
            
            self.simulate_generation()
        
        print("Simulation complete!")
        return self.analyze_results()
    
    def analyze_results(self) -> Dict:
        """Analyze simulation results"""
        analysis = {}
        
        # Detect party flips by measuring position changes
        initial_positions = self.party_history[0]
        final_positions = self.party_history[-1]
        
        position_changes = {}
        for party_key in initial_positions:
            initial = initial_positions[party_key]
            final = final_positions[party_key]
            total_change = np.linalg.norm(final - initial)
            position_changes[party_key] = {
                'total_change': total_change,
                'initial_position': initial,
                'final_position': final,
                'dimension_changes': final - initial
            }
        
        analysis['position_changes'] = position_changes
        analysis['generations'] = len(self.party_history)
        
        # Identify potential flips
        if self.num_parties == 2:
            party_0_change = position_changes['party_0']['dimension_changes']
            party_1_change = position_changes['party_1']['dimension_changes']
            
            # Check if parties moved in opposite directions on key dimensions
            flip_indicators = []
            for dim in range(self.num_dimensions):
                if (party_0_change[dim] > 0.5 and party_1_change[dim] < -0.5) or \
                   (party_0_change[dim] < -0.5 and party_1_change[dim] > 0.5):
                    flip_indicators.append(dim)
            
            analysis['flip_detected'] = len(flip_indicators) >= 2
            analysis['flip_dimensions'] = flip_indicators
        
        return analysis
    
    def plot_results(self):
        """Generate comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Political Party Evolution Simulation Results', fontsize=16)
        
        # 1. Party Position Evolution Over Time
        ax1 = axes[0, 0]
        generations = range(len(self.party_history))
        
        # Plot first two dimensions for visualization
        for party_id in range(self.num_parties):
            dim1_history = [self.party_history[g][f'party_{party_id}'][0] for g in generations]
            dim2_history = [self.party_history[g][f'party_{party_id}'][1] for g in generations]
            
            ax1.plot(dim1_history, dim2_history, label=f'Party {party_id}', linewidth=2, alpha=0.7)
            ax1.scatter(dim1_history[0], dim2_history[0], s=100, marker='o', label=f'P{party_id} Start')
            ax1.scatter(dim1_history[-1], dim2_history[-1], s=100, marker='*', label=f'P{party_id} End')
        
        ax1.set_xlabel('Policy Dimension 1')
        ax1.set_ylabel('Policy Dimension 2')
        ax1.set_title('Party Platform Evolution (2D View)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Voter Distribution Over Time
        ax2 = axes[0, 1]
        for party_id in range(self.num_parties):
            distribution_history = [
                self.voter_distribution_history[g].get(party_id, 0) 
                for g in range(len(self.voter_distribution_history))
            ]
            ax2.plot(distribution_history, label=f'Party {party_id}', linewidth=2)
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Number of Supporters')
        ax2.set_title('Voter Support Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Electoral Fitness Over Time
        ax3 = axes[0, 2]
        for party_id in range(self.num_parties):
            fitness_history = [
                self.election_results_history[g].get(party_id, 0)
                for g in range(len(self.election_results_history))
            ]
            ax3.plot(fitness_history, label=f'Party {party_id}', linewidth=2)
        
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Electoral Fitness')
        ax3.set_title('Party Electoral Success')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Policy Dimension Changes Heatmap
        ax4 = axes[1, 0]
        if self.num_parties <= 5:  # Only show for reasonable number of parties
            dimension_changes = []
            party_labels = []
            
            for party_id in range(self.num_parties):
                initial = self.party_history[0][f'party_{party_id}']
                final = self.party_history[-1][f'party_{party_id}']
                changes = final - initial
                dimension_changes.append(changes)
                party_labels.append(f'Party {party_id}')
            
            dimension_changes = np.array(dimension_changes)
            im = ax4.imshow(dimension_changes, cmap='RdBu_r', aspect='auto')
            ax4.set_xticks(range(self.num_dimensions))
            ax4.set_xticklabels([f'Dim {i}' for i in range(self.num_dimensions)])
            ax4.set_yticks(range(self.num_parties))
            ax4.set_yticklabels(party_labels)
            ax4.set_title('Policy Position Changes by Dimension')
            plt.colorbar(im, ax=ax4, label='Position Change')
        
        # 5. Distance Between Parties Over Time
        ax5 = axes[1, 1]
        if self.num_parties == 2:
            distances = []
            for g in range(len(self.party_history)):
                pos1 = self.party_history[g]['party_0']
                pos2 = self.party_history[g]['party_1']
                distance = np.linalg.norm(pos1 - pos2)
                distances.append(distance)
            
            ax5.plot(distances, linewidth=2, color='purple')
            ax5.set_xlabel('Generation')
            ax5.set_ylabel('Ideological Distance')
            ax5.set_title('Distance Between Parties')
            ax5.grid(True, alpha=0.3)
        
        # 6. Final Party Positions Radar Chart
        ax6 = axes[1, 2]
        if self.num_dimensions <= 8:  # Reasonable for radar chart
            angles = np.linspace(0, 2*np.pi, self.num_dimensions, endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for party_id in range(self.num_parties):
                final_pos = self.party_history[-1][f'party_{party_id}'].tolist()
                final_pos += final_pos[:1]  # Complete the circle
                
                ax6.plot(angles, final_pos, 'o-', linewidth=2, label=f'Party {party_id}')
                ax6.fill(angles, final_pos, alpha=0.25)
            
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels([f'Policy {i}' for i in range(self.num_dimensions)])
            ax6.set_title('Final Party Positions (Radar)')
            ax6.legend()
            ax6.grid(True)
        
        plt.tight_layout()
        plt.show()



######################
# Pre-Main Setup
######################




######################
# Functions
######################

def run_mcmc_analysis(num_runs: int = 50, num_parties: int = 2, generations: int = 200):
    """Run multiple simulations to statistically analyze party flips"""
    print(f"Running MCMC analysis with {num_runs} simulations...")
    
    flip_results = []
    all_analyses = []
    
    for run in range(num_runs):
        if run % 10 == 0:
            print(f"MCMC Run {run}/{num_runs}")
        
        simulator = PoliticalEvolutionSimulator(
            num_parties=num_parties,
            num_voters=10000,
            mutation_rate=random.uniform(0.01, 0.05),  # Vary parameters
            crossover_rate=random.uniform(0.05, 0.15)
        )
        
        analysis = simulator.run_simulation(generations)
        all_analyses.append(analysis)
        
        if num_parties == 2 and analysis.get('flip_detected', False):
            flip_results.append(run)
    
    # Summary statistics
    flip_probability = len(flip_results) / num_runs
    print(f"\nMCMC Analysis Results:")
    print(f"Party flip probability: {flip_probability:.2%}")
    print(f"Number of flips detected: {len(flip_results)}/{num_runs}")
    
    if flip_results:
        print(f"Flips occurred in runs: {flip_results[:10]}{'...' if len(flip_results) > 10 else ''}")
    
    return all_analyses, flip_probability


def main():
    # Single detailed simulation
    print("="*60)
    print("POLITICAL PARTY EVOLUTION GENETIC ALGORITHM")
    print("="*60)
    
    # Run single simulation for detailed analysis
    simulator = PoliticalEvolutionSimulator(
        num_parties=2,
        num_voters=10000,
        num_dimensions=8,
        mutation_rate=0.03,
        crossover_rate=0.1
    )
    
    analysis = simulator.run_simulation(generations=300)
    
    # Display results
    if analysis.get('flip_detected', False):
        print(f"\n🎉 PARTY FLIP DETECTED!")
        print(f"Flip occurred in {len(analysis['flip_dimensions'])} policy dimensions")
    else:
        print(f"\n📊 No major party flip detected in this run")
    
    # Show detailed position changes
    print(f"\nDetailed Position Changes:")
    for party, changes in analysis['position_changes'].items():
        print(f"{party}: Total change = {changes['total_change']:.2f}")
    
    # Generate visualizations
    simulator.plot_results()
    
    # Run MCMC analysis to find flip probability
    print(f"\n" + "="*60)
    print("MCMC STATISTICAL ANALYSIS")
    print("="*60)
    
    mcmc_analyses, flip_prob = run_mcmc_analysis(num_runs=50, generations=200)
    
    print(f"\nStatistical Summary:")
    print(f"Overall flip probability: {flip_prob:.1%}")
    print(f"This suggests that under these conditions, major party")
    print(f"realignments occur in approximately {flip_prob:.1%} of scenarios")


######################
# Main
######################

# The main code call allows this module to be imported as a library or
# called as a standalone program because __name__ will not be properly
# set unless called as a program.

if __name__ == "__main__":
    main()

