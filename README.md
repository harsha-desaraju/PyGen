## PyGen

PyGen is a Python package for optimization using genetic algorithms. Genetic algorithms are search algorithms inspired by evolution. They are particularly useful for finding solutions to optimization and search problems where the search space is large and complex.


## How Genetic Algorithms Work?

The basic idea of a genetic algorithm is to mimic the process of natural selection. The algorithm starts with a population of candidate solutions (often called individuals or chromosomes). Each candidate solution is represented as a string of binary or real-valued numbers, which encodes a potential solution to the problem.

**1. Initialization:** The algorithm starts by creating an initial population of candidate solutions. These solutions are randomly generated or initialized based on some heuristic.

**2. Evaluation:** Each candidate solution in the population is evaluated using a fitness function, which measures how well the solution solves the problem. The fitness function assigns a fitness score to each solution, with higher scores indicating better solutions.

**3. Selection:** The algorithm selects the best-performing solutions (parents) from the current population based on their fitness scores. These solutions are more likely to be selected as parents for the next generation.

**4. Crossover:** The selected parents are combined to create new offspring (children). This is done by exchanging genetic material (genes or parameters) between parents to create new candidate solutions. Crossover helps explore the search space and combine good solutions from different parents.

**5. Mutation:** To maintain genetic diversity in the population and prevent premature convergence, some of the offspring undergo random changes (mutations) in their genetic material. Mutation introduces new genetic material into the population and helps explore new regions of the search space.

**6. Replacement:** The offspring replace some of the least fit members of the current population. This ensures that the population size remains constant and that only the best solutions survive to the next generation.

**7. Termination:** The algorithm terminates when a stopping criterion is met, such as reaching a maximum number of generations or finding a satisfactory solution.

## Why PyGen?

I have been fascinated by genetic algorithms since I first learned about them. I have used them in many problems, but my enthusiasm for genetic algorithms has only grown. So, I decided to write a package from scratch, providing me with an opportunity to deepen my understanding of them and contribute to the open-source community.


Throughout the development process, speed has been a primary focus. To optimize performance, I've extensively vectorized the code and incorporated parallelization when evaluating populations. However, as we're aware, parallelization introduces overhead. Therefore, users have the flexibility to enable or disable parallelization based on the specific requirements of their problem, aiming to enhance the overall program's runtime.




## Getting Started

PyGen is a package still in development. There is a still a lot of functionality to add. But it can still be used. To start using it, place the `pygen` directory in the current working directory and import. One can also add a python path variable to use. Refer to the `example.py` file and the [documentation](https://github.com/harsha-desaraju/PyGen?tab=readme-ov-file#documentation) to understand better.
## Contributing

Contributions are always welcome!


## Documentation

#### Optimizer
The main class in PyGen is the `Optimizer` class, which determines the optimizer based on the problem type. There are three types of optimizers available:

- `BinaryOptimizer`
- `ContinuousOptimizer`
- `DiscreteOptimizer`


#### Selection
The `Selection` class is responsible for choosing solutions from the population as parents for mating. There are several selection policies to choose from, depending on the type of selection required:


- `RandomSelection`
- `RankSelection`
- `RouletteWheelSelection`
- `TournamentSelection`

#### Crossover
The `Crossover` class determines the policy for combining two solutions to generate new solutions (children). There are two types of crossovers supported:


- `UniformCrossover`
- `KPointCrossover`

#### Other parameters
In addition to the above parameters, there are other parameters in the genetic algorithm, such as `selection_frac`, `mu`, `elite_frac`, `cost_function` etc.



