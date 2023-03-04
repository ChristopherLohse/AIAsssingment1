from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
from search import Breadthfirst, Depthfirst, Astar
from mdp import MDPValueIteration, MDPPolicyIteration
import seaborn as sns
import numpy as np
class Mazemanager:
    def __init__(self, maze, entrance, exit):
        self.maze = maze
        self.entrance = entrance
        self.exit = exit
        self.grid = maze.grid
        self.grid[self.entrance[0]][self.entrance[1]] = 0
        self.grid[self.exit[0]][self.exit[1]] =0
    
    def getNeigbours(self, x, y):
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
    
    def showPNG(self, solver, filename = None):
        # taken from mazelib github page
        """Generate a simple image of the maze."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap=plt.cm.binary, interpolation='nearest')
        # show the entrance and exit plot as squares
        plt.plot(self.entrance[1], self.entrance[0], 'gs', label = "entrance")
        plt.plot(self.exit[1], self.exit[0], 'rs', label = "exit")
        plt.xticks([]), plt.yticks([])
        # remove the axis
        plt.axis('off')
        # add the solution path
        for i in range(len(solver.path)-1):
            plt.plot(solver.path[i][1][1], solver.path[i][1][0], 'bo', alpha = 0.5)

        # plot last node
        plt.plot(solver.path[-1][1][1], solver.path[-1][1][0], 'bo', label = "solution path", alpha = 0.5)
        plt.legend(loc = "upper right")
        # tight layout
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
            plt.close()
            return 0
        
        plt.show()

    

if __name__ == "__main__":
    plt.figure(figsize=(8, 8))
    m = Maze()
    m.generator = Prims(5, 5)
    m.generate()
    m.generate_entrances()
    manager = Mazemanager(m, m.start, m.end)
    solver = Breadthfirst(manager)
    solver.solve()
    manager.showPNG(solver = solver, filename = "breadthfirst.png")
    solver2 = Depthfirst(manager)
    solver2.solve()
    manager.showPNG(solver = solver2 , filename = "depthfirst.png")
    solver3 = Astar(manager, heuristic ="manhattan")
    solver3.solve()
    manager.showPNG(solver = solver3, filename = "astar_manhattan.png")
    solver4 = Astar(manager, heuristic ="euclidian")
    solver4.solve()
    manager.showPNG(solver = solver4, filename = "astar_euclidian.png")
    solver5 = MDPValueIteration(manager, discount_factor = 0.99, epsilon = 0.001)
    solver5.solve()
    solver5.plot_utility(filename = "mdp_heatmap.png")
    manager.showPNG(solver = solver5, filename = "mdp.png")
    solver5.plot_policy(filename = "mdp_policy_quiver.png")
    solver6 = MDPPolicyIteration(manager, discount_factor= 0.99, debug=True, epsilon=0.001)
    solver6.solve()
    # show heatmap of utility values
    solver6.plot_utility(filename = "mdp_policy_heatmap.png")
    plt.close()
    solver6.plot_policy(filename = "mdp_policy_policy_quiver.png")
    # calculate ratio time taken for policy iteration to value iteration
    print("time taken for policy iteration: ", solver6.time)
    print("time taken for value iteration: ", solver5.time)
    print("ratio: ", solver6.time/solver5.time)
    manager.showPNG(solver = solver6, filename = "mdp_policy.png")