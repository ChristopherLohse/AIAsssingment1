from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
from search import Breadthfirst, Depthfirst, Astar
from mdp import MDPValueIteration, MDPPolicyIteration
from maze import Mazemanager
import pandas as pd
import tracemalloc

mazesizes = [(5, 5), (10,10), (20,20), (50,50)]

df = pd.DataFrame(columns = ["Maze size", "Algorithm", "Time", "Path length", "Iterations", "Memory"])

size_list = [] 
algorithm_list = []
time_list = []
path_length_list = []
visited_list = []
iterations_list = []
memory_list = []
runs = 3

algorithms = [Breadthfirst, Depthfirst,Astar,MDPValueIteration, MDPPolicyIteration]
names = ["Breadthfirst", "Depthfirst", "Astar", "MDPValueIteration", "MDPPolicyIteration"]
for size in mazesizes:
    for i in range(runs):
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
            if names[i] == "MDPValueIteration" or names[i] == "MDPPolicyIteration":
                solver = algorithm(manager, propabilities =  [1, 0, 0,])
            else:
                solver = algorithm(manager)
            #  measure the memory usage
            tracemalloc.start()
            # solve the maze
            solver.solve()
            # stop measuring the memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            # store the memory usage
            memory_list.append(peak)

            # store the results
            size_list.append(size)
            algorithm_list.append(names[i])
            time_list.append(solver.time)
            
            # check if the algorithm is MDPValueIteration or MDPPolicyIteration
            if names[i] == "MDPValueIteration" or names[i] == "MDPPolicyIteration":
                iterations_list.append(solver.iterations)
                path_length_list.append(solver.steps)
                visited_list.append(0)
            else:
                visited_list.append(solver.steps)
                solver.reversePath()
                path_length_list.append(len(solver.path))
                iterations_list.append(0)

# add the results to the dataframe
df["Maze size"] = size_list
df["Algorithm"] = algorithm_list
df["Time"] = time_list
df["Path length"] = path_length_list
df["Iterations"] = iterations_list
df["Memory"] = memory_list
df["Visited"] = visited_list

# calculate the mean  time for each algorithm and maze size combination same for path length and iterations and memory and visited add standard deviation
df = df.groupby(["Maze size", "Algorithm"]).agg({"Time": ["mean", "std"], "Path length": ["mean", "std"], "Iterations": ["mean", "std"], "Memory": ["mean", "std"], "Visited": ["mean", "std"]})
# save the results to a csv file
df.to_csv("results2.csv")

