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

### Basic Usage
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

## 🛠️ Configuration Options

### Simulation Parameters
```python
simulator = PoliticalEvolutionSimulator(
    num_parties=2,           # Number of political parties
    num_voters=1000,         # Population size
    num_dimensions=8,        # Policy dimensions
    mutation_rate=0.03,      # Rate of platform evolution
    crossover_rate=0.1       # Rate of idea adoption between parties
)
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

### Historical Accuracy
When initialized with Lincoln-era starting positions:
- **Republicans**: Anti-slavery, pro-business, federal power
- **Democrats**: States' rights, rural interests, limited federal power

The model successfully reproduces realignment patterns similar to the historical Lincoln-Reagan transformation.

### Statistical Findings
- Major realignments occur in approximately 15-30% of simulations
- Flip probability increases with higher mutation rates and social change
- Multi-party systems show different stability patterns

## 🔍 Research Applications

### Political Science
- **Party system evolution** analysis
- **Realignment theory** testing
- **Voter behavior** modeling
- **Coalition formation** dynamics

### Computational Social Science
- **Agent-based modeling** of complex systems
- **Evolutionary approaches** to social phenomena
- **Statistical inference** in political processes

### Historical Analysis
- **Counterfactual scenarios** ("What if there were 3 parties?")
- **Parameter estimation** for historical periods
- **Predictive modeling** of future realignments

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

## 🤝 Contributing

We welcome contributions! Areas of particular interest:
- **Historical parameter calibration**
- **Alternative fitness functions**
- **Additional visualization methods**
- **Multi-party system analysis**
- **International political system modeling**

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

## 🎓 Educational Use

This code is well-suited for:
- **Political science courses** on party systems
- **Computer science classes** on genetic algorithms
- **Data science workshops** on agent-based modeling
- **Interdisciplinary research** in computational social science

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for general questions
- **Email**: [phill@moyer.ai](mailto:phil@moyer.ai)

---

*"The only constant in politics is change, and now we can model it."* 🗳️🧬
