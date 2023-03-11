from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
from search import Breadthfirst, Depthfirst, Astar, Search
from mdp import MDPValueIteration, MDPPolicyIteration
import seaborn as sns
import numpy as np
class Mazemanager:
    def __init__(self, maze, entrance, exit):
        self.maze = maze
        self.entrance = entrance
        self.exit = exit
        self.grid = maze.grid
        # this is to create more than one path to the exit
        for x in range(1, self.grid.shape[0] -1):
                for y in range(1, self.grid.shape[1] -1):
                    if np.random.random() < 0.1:
                        # check if it is not surrounded by walls
                        if self.grid[x-1][y] == 0 and self.grid[x+1][y] == 0:
                            self.grid[x][y] = 0
                        if self.grid[x][y-1] == 0 and self.grid[x][y+1] == 0:
                            self.grid[x][y] = 0
        self.grid[self.entrance[0]][self.entrance[1]] = 0
        self.grid[self.exit[0]][self.exit[1]] =0
        
        self.maze.grid = self.grid
    
    def getNeigbours(self, x, y, fill = False):
        neighbours = []
        # check left before checking if out of bounds
        if x > 0 and self.grid[x-1][y] == 0:
            neighbours.append((x-1, y))
        # check right
        if x < self.grid.shape[0]-1 and self.grid[x+1][y] == 0:
            neighbours.append((x+1, y))
        # check up
        if y > 0 and self.grid[x][y-1] == 0:
            neighbours.append((x, y-1))
        # check down
        if y < self.grid.shape[1]-1 and self.grid[x][y+1] == 0:
            neighbours.append((x, y+1))
        return neighbours
    
    def getNeigbours_mdp(self, x, y):
        neighbours = []
        # check left before checking if out of bounds
        if x > 0:
            # check if wall
            if self.grid[x-1][y] == 0:
                neighbours.append((x-1, y))
            else:
                neighbours.append((x, y))

        # check right
        if x < self.grid.shape[0]-1:
            if self.grid[x+1][y] == 0:
                neighbours.append((x+1, y))
            else:
                neighbours.append((x, y))
        # check up
        if y > 0:
            if self.grid[x][y-1] == 0:
                neighbours.append((x, y-1))
            else:
                neighbours.append((x, y))
        # check down
        if y < self.grid.shape[1]-1:
            if self.grid[x][y+1] == 0:
                neighbours.append((x, y+1))
            else:
                neighbours.append((x, y))

        return neighbours
            
    def showPNG(self, solver, filename = None):
        # taken from mazelib github page
        """Generate a simple image of the maze."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap=plt.cm.binary, interpolation='nearest')
        # show the entrance and exit plot as squares
        
        plt.xticks([]), plt.yticks([])
        # remove the axis
        plt.axis('off')
        # add the solution path
        
        # check if the type is search#
        if type(solver) == Breadthfirst or type(solver) == Depthfirst or type(solver) == Astar:
            for i in range(1, len(solver.path)-2):
                plt.plot(solver.path[i][1][1], solver.path[i][1][0], 'bo', alpha = 0.5, color = "purple")
            # reverse the path
            solver.reversePath()
            # plot the path
            for i in range(len(solver.path)):
                plt.plot(solver.path[i][1][1], solver.path[i][1][0], 'bo', alpha = 1)
            plt.plot(solver.path[-2][1][1], solver.path[-2][1][0], 'bo', label = "solution path", alpha = 1)
            plt.plot(solver.path[-2][1][1], solver.path[-2][1][0], 'bo', label = "explored nodes", alpha = 0.5, color = "purple")
        else:
            for i in range(len(solver.path)):
                plt.plot(solver.path[i][1][1], solver.path[i][1][0], 'bo', alpha = 1)
            plt.plot(solver.path[1][1][1], solver.path[1][1][0], 'bo', label = "solution path", alpha = 1)

        plt.plot(self.entrance[1], self.entrance[0], 'gs', label = "entrance")
        plt.plot(self.exit[1], self.exit[0], 'rs', label = "exit")
        
        plt.legend(loc = "upper left", prop={'size': 20})
        # tight layout
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
            plt.close()
            return 0
        
        plt.show()

    

if __name__ == "__main__":
    plt.figure(figsize=(8, 8))
    m = Maze(21)
    m.generator = Prims(10, 10)

  

    m.generate()
    m.generate_entrances()

    manager = Mazemanager(m, m.start, m.end)
    print("size of the maze: ", manager.grid.shape)
    solver = Breadthfirst(manager)
    solver.solve()
    manager.showPNG(solver = solver, filename = "../Figures/breadthfirst.png")
    solver2 = Depthfirst(manager)
    solver2.solve()
    manager.showPNG(solver = solver2 , filename = "../Figures/depthfirst.png")
    solver3 = Astar(manager, heuristic ="manhattan")
    solver3.solve()
    manager.showPNG(solver = solver3, filename = "../Figures/astar_manhattan.png")
    solver4 = Astar(manager, heuristic ="euclidian")
    solver4.solve()
    manager.showPNG(solver = solver4, filename = "../Figures/astar_euclidian.png")
    solver5 = MDPValueIteration(manager, discount_factor = 0.9, epsilon = 0.000000001, debug = False, propabilities = [0.8, 0.1, 0.1])
    solver5.solve()
    solver5.plot_utility(filename = "../Figures/mdp_heatmap.png")
    manager.showPNG(solver = solver5, filename = "../Figures/mdp.png")
    solver5.plot_policy(filename = "../Figures/mdp_policy_quiver.png")
    solver6 = MDPPolicyIteration(manager, discount_factor= 0.9, debug=False, epsilon=0.000001, propabilities = [0.8, 0.1, 0.1])
    solver6.solve()
    solver6.plot_policy(filename = "../Figures/mdp_policy_policy_quiver.png")
    # calculate ratio time taken for policy iteration to value iteration
    print("time taken for policy iteration: ", solver6.time)
    print("time taken for value iteration: ", solver5.time)
    print("ratio: ", solver6.time/solver5.time)
    manager.showPNG(solver = solver6, filename = "../Figures/mdp_policy.png")