# Political Party Evolution Genetic Algorithm

A computational model that simulates the evolution of political parties and voter preferences over time using genetic algorithms, with a focus on understanding major party realignments like the famous Lincoln-Reagan era flip in American politics.

## 🎯 Overview

This project models political systems as evolutionary processes where:
- **Political parties** adapt their platforms based on electoral success
- **Voters** switch allegiances when their preferences diverge from their current party
- **Major realignments** emerge naturally from the interaction of these dynamics

The model uses genetic algorithms to simulate how parties evolve their positions across multiple policy dimensions, while Monte Carlo methods help quantify the probability of major political realignments.

## 🔬 Scientific Approach

### Genetic Algorithm Components
- **Genotype**: Party platforms represented as n-dimensional vectors
- **Phenotype**: Electoral success and voter support
- **Mutation**: Gradual policy position shifts
- **Crossover**: Parties adopting successful positions from competitors
- **Selection**: Electoral pressure drives platform evolution

### Agent-Based Modeling
- **Voter Agents**: Individual preferences, party loyalty, switching behavior
- **Party Agents**: Platform positions, adaptation mechanisms
- **Environmental Pressure**: Electoral cycles, generational change

### Statistical Analysis
- **MCMC Simulation**: Multiple runs with parameter variation
- **Flip Detection**: Algorithmic identification of major realignments
- **Probability Estimation**: Statistical quantification of realignment likelihood

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy matplotlib seaborn pandas
```

### Command Line Interface

The easiest way to run simulations is through the command-line interface:

#### Quick Test Run
```bash
python political_cli.py quick
```

#### Single Detailed Simulation
```bash
python political_cli.py single --generations 500 --num-voters 15000
```

#### Statistical Analysis (MCMC)
```bash
python political_cli.py mcmc --mcmc-runs 100 --generations 200
```

#### Custom Parameters
```bash
python political_cli.py single \
    --num-parties 3 \
    --num-dimensions 10 \
    --mutation-rate 0.08 \
    --crossover-rate 0.15 \
    --output-dir ./my_analysis
```

### Programmatic Usage
```python
from political_evolution_ga import PoliticalEvolutionSimulator

# Create simulator with default 2-party system
simulator = PoliticalEvolutionSimulator(
    num_parties=2,
    num_voters=1000,
    num_dimensions=8,
    mutation_rate=0.03
)

# Run simulation
analysis = simulator.run_simulation(generations=300)

# Generate visualizations
simulator.plot_results()

# Check for party flip
if analysis.get('flip_detected', False):
    print("Major party realignment detected!")
```

### MCMC Statistical Analysis
```python
from political_evolution_ga import run_mcmc_analysis

# Run multiple simulations to estimate flip probability
analyses, flip_probability = run_mcmc_analysis(
    num_runs=50, 
    num_parties=2, 
    generations=200
)

print(f"Party flip probability: {flip_probability:.1%}")
```

## 📊 Features

### Core Simulation
- **Multi-dimensional policy space** (configurable dimensions)
- **Realistic voter behavior** with loyalty and switching dynamics
- **Party evolution** through mutation and crossover
- **Generational change** in voter preferences
- **Electoral feedback loops**

### Analysis Tools
- **Flip detection algorithm** identifies major realignments
- **Statistical significance testing** via MCMC methods
- **Parameter sensitivity analysis**
- **Historical trajectory tracking**

### Visualization Suite
- **Party trajectory plots** in 2D policy space
- **Voter support evolution** over time
- **Electoral fitness tracking**
- **Policy dimension heatmaps**
- **Ideological distance measurements**
- **Radar charts** for final positions

### Command Line Interface
- **Three analysis modes**: `single`, `mcmc`, and `quick`
- **Automatic file management** with timestamped output directories
- **High-resolution plots** saved as PNG files (300 DPI)
- **JSON data export** for web integration
- **Human-readable summary reports**
- **Full parameter control** via command-line arguments

## 📁 Output Structure

Each simulation run creates a timestamped directory containing:

```
./output/run_20250611_143022/
├── simulation_analysis.png     # Complete visualization suite
├── results.json               # Machine-readable data
└── summary_report.txt         # Human-readable analysis
```

### Visualization Output
Your results include six key visualizations:
- **Party Platform Evolution**: Trajectories through ideological space
- **Voter Support Over Time**: Electoral dynamics
- **Electoral Fitness**: Party adaptation success
- **Policy Dimension Changes**: Heatmap of position shifts
- **Distance Between Parties**: Polarization/convergence trends
- **Final Party Positions**: Radar chart of end states

### JSON Data Format
Perfect for web applications and further analysis:
```json
{
  "simulation_parameters": {...},
  "single_run_analysis": {...},
  "mcmc_analysis": {...},
  "timestamp": "2025-06-11T14:30:22",
  "version": "1.0"
}
```

## 🛠️ CLI Configuration Options

### Single Simulation Mode
```bash
python political_cli.py single [options]

Options:
  --num-parties INT         Number of political parties (default: 2)
  --num-voters INT          Number of voters (default: 10000)
  --num-dimensions INT      Policy dimensions (default: 8)
  --mutation-rate FLOAT     Platform mutation rate (default: 0.06)
  --crossover-rate FLOAT    Platform crossover rate (default: 0.2)
  --generations INT         Simulation length (default: 300)
  --output-dir PATH         Output directory (default: ./output)
```

### MCMC Analysis Mode
```bash
python political_cli.py mcmc [options]

Additional MCMC Options:
  --mcmc-runs INT              Number of simulation runs (default: 50)
  --mutation-rate-min FLOAT    Min mutation rate for variation (default: 0.04)
  --mutation-rate-max FLOAT    Max mutation rate for variation (default: 0.07)
  --crossover-rate-min FLOAT   Min crossover rate (default: 0.15)
  --crossover-rate-max FLOAT   Max crossover rate (default: 0.25)
  --save-all-runs             Save detailed results from all runs
```

### Policy Dimensions
The default 8-dimensional model represents:
1. **Economic Policy** (free market ↔ government intervention)
2. **Social Issues** (traditional ↔ progressive values)
3. **Federal Power** (states' rights ↔ federal authority)
4. **Civil Rights** (restrictive ↔ expansive)
5. **Foreign Policy** (isolationist ↔ interventionist)
6. **Environmental Policy** (development ↔ conservation)
7. **Immigration** (restrictive ↔ open)
8. **Government Size** (limited ↔ expansive)

## 📈 Example Results

### Command Line Usage
```bash
# Quick test
$ python political_cli.py quick
Running single simulation...
🎉 PARTY FLIP DETECTED!
Flip occurred in 3 policy dimensions
✅ Analysis complete! Results saved to: ./output/run_20250611_143022

# Statistical analysis
$ python political_cli.py mcmc --mcmc-runs 100
MCMC Analysis Results:
Overall flip probability: 23.0%
Number of flips detected: 23/100
✅ Analysis complete! Results saved to: ./output/run_20250611_143856
```

### Historical Accuracy
When initialized with Lincoln-era starting positions:
- **Republicans**: Anti-slavery, pro-business, federal power
- **Democrats**: States' rights, rural interests, limited federal power

The model successfully reproduces realignment patterns similar to the historical Lincoln-Reagan transformation.

### Statistical Findings
- Major realignments occur in approximately 15-30% of simulations under default parameters
- Flip probability increases with higher mutation rates and social change
- Multi-party systems (3+ parties) show dramatically different stability patterns
- Party flips are rare in multi-party systems due to coalition dynamics

## 🔍 Research Applications

### Political Science
- **Party system evolution** analysis
- **Realignment theory** testing
- **Voter behavior** modeling
- **Coalition formation** dynamics
- **Electoral system impact** studies

### Computational Social Science
- **Agent-based modeling** of complex systems
- **Evolutionary approaches** to social phenomena
- **Statistical inference** in political processes
- **Parameter sensitivity analysis**

### Historical Analysis
- **Counterfactual scenarios** ("What if there were 3 parties?")
- **Parameter estimation** for historical periods
- **Predictive modeling** of future realignments
- **Cross-national comparisons**

### Web Integration
The JSON output format makes integration with web applications straightforward:
- **Real-time parameter adjustment** via web interfaces
- **Interactive visualization** of results
- **Batch processing** of multiple scenarios
- **RESTful API** development for simulation services

## 📚 Algorithm Details

### Fitness Function
Party fitness combines:
- **Electoral support** (percentage of voter base)
- **Representation quality** (how well party represents supporters)

### Mutation Mechanism
- **Gaussian perturbation** of policy positions
- **Environmental pressure** modulation
- **Bounded policy space** (-2 to +2 range)

### Voter Switching Model
Probability of party switching:
```
P(switch) = (1 - loyalty) × (current_distance - best_distance)
```

### Flip Detection
Algorithmic identification based on:
- **Magnitude of position changes** (>0.5 standard deviations)
- **Opposite direction movement** between parties
- **Multiple dimension involvement** (≥2 dimensions)

## 🎯 Interpreting Results

### Flip Probability Ranges
| Probability | Interpretation | Real-World Analogy |
|------------|----------------|-------------------|
| 0-5% | Very stable system | Post-WWII consensus era |
| 5-15% | Moderate instability | Normal democratic competition |
| 15-30% | High realignment potential | Crisis periods (1930s, 1960s) |
| 30%+ | Highly unstable system | Revolutionary periods |

### Visual Indicators
- **Crossing trajectories**: Parties swapping ideological positions
- **Voter support shifts**: Rapid changes indicate realignment events
- **Distance oscillations**: Polarization and convergence cycles
- **Policy heatmaps**: Red/blue patterns show directional changes

See the included `Interpreting_Your_Results.md` guide for detailed analysis instructions.

## 🤝 Contributing

We welcome contributions! Areas of particular interest:
- **Historical parameter calibration**
- **Alternative fitness functions**
- **Additional visualization methods**
- **Multi-party system analysis**
- **International political system modeling**
- **Web interface development**

### Development Setup
```bash
git clone https://github.com/MoyerScientific/political-evolution-ga.git
cd political-evolution-ga
pip install -r requirements.txt
python -m pytest tests/
```

## 📖 Citation

If you use this code in academic research, please cite:
```bibtex
@software{political_evolution_ga,
  title = {Political Party Evolution Genetic Algorithm},
  author = {Philip Moyer},
  year = {2025},
  url = {https://github.com/MoyerScientific/political-evolution-ga}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- **Evolutionary Game Theory** in political science
- **Agent-based models** of voting behavior
- **Computational political economy**
- **Social choice theory** and genetic algorithms
- **Duverger's Law** and party system dynamics

## 🎓 Educational Use

This code is well-suited for:
- **Political science courses** on party systems and realignment theory
- **Computer science classes** on genetic algorithms and agent-based modeling
- **Data science workshops** on statistical simulation and MCMC methods
- **Interdisciplinary research** in computational social science
- **Public policy analysis** and electoral system design

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for general questions
- **Email**: [phil@moyer.ai](mailto:phil@moyer.ai)

---

*"Democracy is not just the art of the possible, but the science of the probable."* 🗳️🧬