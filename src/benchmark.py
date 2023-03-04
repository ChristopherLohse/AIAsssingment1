from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
from search import Breadthfirst, Depthfirst, Astar
from mdp import MDPValueIteration, MDPPolicyIteration
from maze import Mazemanager
import pandas as pd

mazesizes = [(5,5), (10,10), (30, 30)]

df = pd.DataFrame(columns = ["Maze size", "Algorithm", "Time", "Path length", "Iterations"])

size_list = [] 
algorithm_list = []
time_list = []
path_length_list = []
iterations_list = []

algorithms = [Breadthfirst, Depthfirst, Astar, MDPValueIteration, MDPPolicyIteration]
names = ["Breadthfirst", "Depthfirst", "Astar", "MDPValueIteration", "MDPPolicyIteration"]
for size in mazesizes:
    # perform 10 runs for each maze size
    for i in range(10):
        # generate a maze
        m = Maze()
        m.generator = Prims(size[0], size[1])
        m.generate()
        m.generate_entrances()
        entrance = m.start
        exit = m.end
        manager = Mazemanager(m, entrance, exit)
        # solve the maze with each algorithm
        for i, algorithm in enumerate(algorithms):
            # create an instance of the algorithm
            solver = algorithm(manager)
            # solve the maze
            solver.solve()
            # store the results
            size_list.append(size)
            algorithm_list.append(names[i])
            time_list.append(solver.time)
            path_length_list.append(solver.steps)
            # check if the algorithm is MDPValueIteration or MDPPolicyIteration
            if names[i] == "MDPValueIteration" or names[i] == "MDPPolicyIteration":
                iterations_list.append(solver.iterations)
            else:
                iterations_list.append(0)

# add the results to the dataframe
df["Maze size"] = size_list
df["Algorithm"] = algorithm_list
df["Time"] = time_list
df["Path length"] = path_length_list
df["Iterations"] = iterations_list
# calculate the average time and path length for each algorithm with standard deviation
df = df.groupby(["Maze size", "Algorithm"]).agg({"Time": ["mean", "std"], "Path length": ["mean", "std"], "Iterations": ["mean", "std"]})
print(df)
# save the results to a csv file
df.to_csv("results.csv")
