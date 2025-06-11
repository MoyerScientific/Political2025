#!/usr/bin/env python3
###############################################################################
# Political Evolution CLI - Command line interface for political party analysis
#
# This module provides a command-line interface for running political party
# evolution simulations with customizable parameters.
#
# Author:      Phil Moyer (phil@moyer.ai)
# Date:        June 2025
# Copyright(c) 2025 Philip R. Moyer. All rights reserved.
#
###############################################################################

import argparse
import os
import sys
import json
import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt

# Import the main simulation module
from political_evolution_ga import PoliticalEvolutionSimulator, run_mcmc_analysis


def setup_output_directory(output_dir: str = "./output") -> Path:
    """Create output directory if it doesn't exist"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create timestamped subdirectory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    return run_dir


def save_simulation_plots(simulator, output_dir: Path, prefix: str = "simulation"):
    """Save all simulation plots to files"""
    print(f"Saving plots to {output_dir}")
    
    # Generate the plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Political Party Evolution Simulation Results', fontsize=16)
    
    # Copy the plotting logic from the original plot_results method
    generations = range(len(simulator.party_history))
    
    # 1. Party Position Evolution Over Time
    ax1 = axes[0, 0]
    for party_id in range(simulator.num_parties):
        dim1_history = [simulator.party_history[g][f'party_{party_id}'][0] for g in generations]
        dim2_history = [simulator.party_history[g][f'party_{party_id}'][1] for g in generations]
        
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
    for party_id in range(simulator.num_parties):
        distribution_history = [
            simulator.voter_distribution_history[g].get(party_id, 0) 
            for g in range(len(simulator.voter_distribution_history))
        ]
        ax2.plot(distribution_history, label=f'Party {party_id}', linewidth=2)
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Number of Supporters')
    ax2.set_title('Voter Support Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Electoral Fitness Over Time
    ax3 = axes[0, 2]
    for party_id in range(simulator.num_parties):
        fitness_history = [
            simulator.election_results_history[g].get(party_id, 0)
            for g in range(len(simulator.election_results_history))
        ]
        ax3.plot(fitness_history, label=f'Party {party_id}', linewidth=2)
    
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Electoral Fitness')
    ax3.set_title('Party Electoral Success')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Policy Dimension Changes Heatmap
    ax4 = axes[1, 0]
    if simulator.num_parties <= 5:
        import numpy as np
        dimension_changes = []
        party_labels = []
        
        for party_id in range(simulator.num_parties):
            initial = simulator.party_history[0][f'party_{party_id}']
            final = simulator.party_history[-1][f'party_{party_id}']
            changes = final - initial
            dimension_changes.append(changes)
            party_labels.append(f'Party {party_id}')
        
        dimension_changes = np.array(dimension_changes)
        im = ax4.imshow(dimension_changes, cmap='RdBu_r', aspect='auto')
        ax4.set_xticks(range(simulator.num_dimensions))
        ax4.set_xticklabels([f'Dim {i}' for i in range(simulator.num_dimensions)])
        ax4.set_yticks(range(simulator.num_parties))
        ax4.set_yticklabels(party_labels)
        ax4.set_title('Policy Position Changes by Dimension')
        plt.colorbar(im, ax=ax4, label='Position Change')
    
    # 5. Distance Between Parties Over Time
    ax5 = axes[1, 1]
    if simulator.num_parties == 2:
        import numpy as np
        distances = []
        for g in range(len(simulator.party_history)):
            pos1 = simulator.party_history[g]['party_0']
            pos2 = simulator.party_history[g]['party_1']
            distance = np.linalg.norm(pos1 - pos2)
            distances.append(distance)
        
        ax5.plot(distances, linewidth=2, color='purple')
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Ideological Distance')
        ax5.set_title('Distance Between Parties')
        ax5.grid(True, alpha=0.3)
    
    # 6. Final Party Positions Radar Chart
    ax6 = axes[1, 2]
    if simulator.num_dimensions <= 8:
        import numpy as np
        angles = np.linspace(0, 2*np.pi, simulator.num_dimensions, endpoint=False).tolist()
        angles += angles[:1]
        
        for party_id in range(simulator.num_parties):
            final_pos = simulator.party_history[-1][f'party_{party_id}'].tolist()
            final_pos += final_pos[:1]
            
            ax6.plot(angles, final_pos, 'o-', linewidth=2, label=f'Party {party_id}')
            ax6.fill(angles, final_pos, alpha=0.25)
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels([f'Policy {i}' for i in range(simulator.num_dimensions)])
        ax6.set_title('Final Party Positions (Radar)')
        ax6.legend()
        ax6.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_dir / f"{prefix}_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def save_results_json(analysis: dict, mcmc_results: dict, output_dir: Path, 
                     simulation_params: dict, prefix: str = "results"):
    """Save simulation results as JSON"""
    results = {
        'simulation_parameters': simulation_params,
        'single_run_analysis': analysis,
        'mcmc_analysis': mcmc_results,
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '1.0'
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results = convert_numpy(results)
    
    json_file = output_dir / f"{prefix}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return json_file


def create_summary_report(analysis: dict, mcmc_results: dict, output_dir: Path,
                         simulation_params: dict):
    """Create a human-readable summary report"""
    report_file = output_dir / "summary_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("POLITICAL PARTY EVOLUTION SIMULATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SIMULATION PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        for key, value in simulation_params.items():
            f.write(f"{key:20}: {value}\n")
        f.write("\n")
        
        f.write("SINGLE SIMULATION RESULTS:\n")
        f.write("-" * 30 + "\n")
        if analysis.get('flip_detected', False):
            f.write("🎉 PARTY FLIP DETECTED!\n")
            f.write(f"Flip occurred in {len(analysis.get('flip_dimensions', []))} policy dimensions\n")
            f.write(f"Flip dimensions: {analysis.get('flip_dimensions', [])}\n")
        else:
            f.write("📊 No major party flip detected in this run\n")
        f.write("\n")
        
        f.write("Position Changes by Party:\n")
        for party, changes in analysis.get('position_changes', {}).items():
            f.write(f"  {party}: Total change = {changes.get('total_change', 0):.3f}\n")
        f.write("\n")
        
        if mcmc_results:
            f.write("MCMC STATISTICAL ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            flip_prob = mcmc_results.get('flip_probability', 0)
            num_runs = mcmc_results.get('num_runs', 0)
            num_flips = mcmc_results.get('num_flips', 0)
            
            f.write(f"Number of simulation runs: {num_runs}\n")
            f.write(f"Number of flips detected: {num_flips}\n")
            f.write(f"Overall flip probability: {flip_prob:.1%}\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write(f"Under the specified conditions, major party realignments\n")
            f.write(f"occur in approximately {flip_prob:.1%} of scenarios.\n")
            
            if flip_prob > 0.15:
                f.write("This suggests relatively high instability in party positions.\n")
            elif flip_prob > 0.05:
                f.write("This suggests moderate potential for party realignment.\n")
            else:
                f.write("This suggests relatively stable party positions over time.\n")
    
    return report_file


def run_single_simulation(args):
    """Run a single detailed simulation"""
    print("Running single simulation...")
    
    # Create output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Set up simulation parameters
    sim_params = {
        'num_parties': args.num_parties,
        'num_voters': args.num_voters,
        'num_dimensions': args.num_dimensions,
        'mutation_rate': args.mutation_rate,
        'crossover_rate': args.crossover_rate,
        'generations': args.generations
    }
    
    # Run simulation
    simulator = PoliticalEvolutionSimulator(
        num_parties=args.num_parties,
        num_voters=args.num_voters,
        num_dimensions=args.num_dimensions,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate
    )
    
    analysis = simulator.run_simulation(generations=args.generations)
    
    # Save results
    plot_file = save_simulation_plots(simulator, output_dir)
    json_file = save_results_json(analysis, {}, output_dir, sim_params)
    report_file = create_summary_report(analysis, {}, output_dir, sim_params)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Plot: {plot_file.name}")
    print(f"  - Data: {json_file.name}")
    print(f"  - Report: {report_file.name}")
    
    # Display key results
    if analysis.get('flip_detected', False):
        print(f"\n🎉 PARTY FLIP DETECTED!")
        print(f"Flip occurred in {len(analysis['flip_dimensions'])} policy dimensions")
    else:
        print(f"\n📊 No major party flip detected in this run")
    
    return output_dir


def run_mcmc_simulation(args):
    """Run MCMC analysis with multiple simulations"""
    print("Running MCMC analysis...")
    
    # Create output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Set up simulation parameters
    sim_params = {
        'num_parties': args.num_parties,
        'num_voters': args.num_voters,
        'num_dimensions': args.num_dimensions,
        'mutation_rate_range': [args.mutation_rate_min, args.mutation_rate_max],
        'crossover_rate_range': [args.crossover_rate_min, args.crossover_rate_max],
        'generations': args.generations,
        'mcmc_runs': args.mcmc_runs
    }
    
    # Run MCMC analysis
    all_analyses, flip_probability = run_mcmc_analysis(
        num_runs=args.mcmc_runs,
        num_parties=args.num_parties,
        generations=args.generations
    )
    
    # Compile MCMC results
    mcmc_results = {
        'flip_probability': flip_probability,
        'num_runs': args.mcmc_runs,
        'num_flips': int(flip_probability * args.mcmc_runs),
        'all_analyses': all_analyses[:10] if args.save_all_runs else []  # Limit to save space
    }
    
    # Also run one detailed simulation for plotting
    simulator = PoliticalEvolutionSimulator(
        num_parties=args.num_parties,
        num_voters=args.num_voters,
        num_dimensions=args.num_dimensions,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate
    )
    single_analysis = simulator.run_simulation(generations=args.generations)
    
    # Save results
    plot_file = save_simulation_plots(simulator, output_dir, "mcmc")
    json_file = save_results_json(single_analysis, mcmc_results, output_dir, sim_params, "mcmc_results")
    report_file = create_summary_report(single_analysis, mcmc_results, output_dir, sim_params)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Plot: {plot_file.name}")
    print(f"  - Data: {json_file.name}")
    print(f"  - Report: {report_file.name}")
    
    # Display key results
    print(f"\nMCMC Analysis Results:")
    print(f"Overall flip probability: {flip_probability:.1%}")
    print(f"Number of flips detected: {mcmc_results['num_flips']}/{args.mcmc_runs}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Political Party Evolution Simulator - Analyze party flip probabilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Subcommands for different types of analysis
    subparsers = parser.add_subparsers(dest='command', help='Analysis type')
    
    # Single simulation command
    single_parser = subparsers.add_parser('single', help='Run single detailed simulation')
    single_parser.add_argument('--num-parties', type=int, default=2,
                              help='Number of political parties')
    single_parser.add_argument('--num-voters', type=int, default=10000,
                              help='Number of voters in simulation')
    single_parser.add_argument('--num-dimensions', type=int, default=8,
                              help='Number of policy dimensions')
    single_parser.add_argument('--mutation-rate', type=float, default=0.06,
                              help='Party platform mutation rate')
    single_parser.add_argument('--crossover-rate', type=float, default=0.2,
                              help='Party platform crossover rate')
    single_parser.add_argument('--generations', type=int, default=300,
                              help='Number of generations to simulate')
    single_parser.add_argument('--output-dir', type=str, default='./output',
                              help='Output directory for results')
    
    # MCMC analysis command
    mcmc_parser = subparsers.add_parser('mcmc', help='Run MCMC statistical analysis')
    mcmc_parser.add_argument('--num-parties', type=int, default=2,
                            help='Number of political parties')
    mcmc_parser.add_argument('--num-voters', type=int, default=10000,
                            help='Number of voters in simulation')
    mcmc_parser.add_argument('--num-dimensions', type=int, default=8,
                            help='Number of policy dimensions')
    mcmc_parser.add_argument('--mutation-rate', type=float, default=0.06,
                            help='Base mutation rate (for single example)')
    mcmc_parser.add_argument('--crossover-rate', type=float, default=0.2,
                            help='Base crossover rate (for single example)')
    mcmc_parser.add_argument('--mutation-rate-min', type=float, default=0.04,
                            help='Minimum mutation rate for MCMC')
    mcmc_parser.add_argument('--mutation-rate-max', type=float, default=0.07,
                            help='Maximum mutation rate for MCMC')
    mcmc_parser.add_argument('--crossover-rate-min', type=float, default=0.15,
                            help='Minimum crossover rate for MCMC')
    mcmc_parser.add_argument('--crossover-rate-max', type=float, default=0.25,
                            help='Maximum crossover rate for MCMC')
    mcmc_parser.add_argument('--generations', type=int, default=300,
                            help='Number of generations per simulation')
    mcmc_parser.add_argument('--mcmc-runs', type=int, default=50,
                            help='Number of MCMC simulation runs')
    mcmc_parser.add_argument('--output-dir', type=str, default='./output',
                            help='Output directory for results')
    mcmc_parser.add_argument('--save-all-runs', action='store_true',
                            help='Save detailed results from all MCMC runs')
    
    # Quick test command
    quick_parser = subparsers.add_parser('quick', help='Quick test with minimal parameters')
    quick_parser.add_argument('--output-dir', type=str, default='./output',
                             help='Output directory for results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'single':
            output_dir = run_single_simulation(args)
        elif args.command == 'mcmc':
            output_dir = run_mcmc_simulation(args)
        elif args.command == 'quick':
            # Quick test with minimal parameters
            args.num_parties = 2
            args.num_voters = 1000
            args.num_dimensions = 4
            args.mutation_rate = 0.06
            args.crossover_rate = 0.2
            args.generations = 50
            output_dir = run_single_simulation(args)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
        print(f"📁 You can now examine the results in the output directory.")
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

