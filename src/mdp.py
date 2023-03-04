import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)

class MDP:
    def __init__(self, mazemanager, discount_factor=0.99, rs= 0):
        self.mazemanager = mazemanager
        self.grid = mazemanager.grid
        self.entrance = mazemanager.entrance
        self.exit = mazemanager.exit
        self.values = np.zeros(self.grid.shape)
        self.values[self.exit[0]][self.exit[1]] = 1
        self.values[self.values == 0] = rs
        self.setWalls()
        self.discount_factor = discount_factor
        self.utilitys = np.zeros(self.grid.shape)
        self.path = []
        self.policy = np.zeros(shape = (self.grid.shape[0], self.grid.shape[1], 2))
        self.start_time = datetime.now()
        self.iterations = 0

    def setWalls(self):
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x][y] == 1:
                    self.values[x][y] = None # set walls to none to ignore them 


    def calculateutilitycell(self, x, y, utilitys):
        utility = 0
        neighbours = self.mazemanager.getNeigbours(x, y)
        # check if exit
        if (x, y) == self.exit:
            return 1, None
        # take the max of the utilities of the neighbours following the Bellamann equation
        neighbour_index = neighbours[0]
        for i, neighbour in enumerate(neighbours):
            if utilitys[neighbour[0]][neighbour[1]] > utility:
                utility = utilitys[neighbour[0]][neighbour[1]]
                neighbour_index = neighbours[i]
        direction_x = int(neighbour_index[0] - x)
        direction_y = int(neighbour_index[1] - y)
        direction = np.array([direction_x, direction_y])
        utility += self.values[x][y]
        utility *= self.discount_factor 
        return utility, direction
    

    def findpath(self):
        current = self.entrance
        while True:
            if current == self.exit:
                return
            nextcell_x = int(current[0] + self.policy[current[0]][current[1]][0])
            nextcell_y = int(current[1] + self.policy[current[0]][current[1]][1])
            self.path.append((current, (nextcell_x, nextcell_y)))
            current = nextcell_x, nextcell_y
    
    def plot_policy(self, filename=None, internal = False):
         # plot vector of policy values as quiver plot
        plt.figure(figsize=(10, 10))
        # add the grid
        plt.imshow(self.grid, cmap=plt.cm.binary, interpolation='nearest')
        # show the entrance and exit plot as squares
        plt.plot(self.entrance[1], self.entrance[0], 'gs', label = "Entrance")
        plt.plot(self.exit[1], self.exit[0], 'rs', label = "Exit")
        plt.legend(loc = "upper right")
        plt.xticks([]), plt.yticks([])
        plt.axis('off')
        # add the vector plot of the policy
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] != 1:
                    current = (i, j)
                    if np.isnan(self.policy[current[0]][current[1]][0]):
                        continue
                    # plot arrow slightly smaller than cell
                    plt.arrow(current[1], current[0], self.policy[current[0]][current[1]][1]*0.5, self.policy[current[0]][current[1]][0]*0.5, head_width=0.1, head_length=0.1, fc='k', ec='k')
                    # add the utility value to the cell next to the arrow in red
                    plt.text(current[1] + self.policy[current[0]][current[1]][1]*0.5, current[0] + self.policy[current[0]][current[1]][0]*0.5, round(self.utilitys[current[0]][current[1]], 3), color='r')
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            plt.close()
            return
        else:
            plt.show()

    def plot_utility(self, filename=None):
        values = self.utilitys
        # replace 0 with nan
        values[values == 0] = np.nan
        # change plot size
        plt.figure(figsize=(10, 10))
        sns.heatmap(values, annot = True, cbar=False)
        # remove ticks
        plt.xticks([]), plt.yticks([])
        plt.axis('off')

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            plt.close()
            
        else:
            plt.show()
        
class MDPValueIteration(MDP):
    def __init__(self, mazemanager, discount_factor=0.99, epsilon=0.1, rs= 0):
        super().__init__(mazemanager, discount_factor, rs)
        self.epsilon = epsilon
        self.deltas = []

    def valueiteration(self):
        utilitys_temp = np.zeros(self.grid.shape)
        i = 0
        while True:
            i += 1
            utilitys = np.copy(utilitys_temp)
            delta = 0
            for x in range(self.grid.shape[0]):
                for y in range(self.grid.shape[1]):
                    if np.isnan(self.values[x][y]) == False:
                        utilitys_temp[x][y], self.policy[x][y]  = self.calculateutilitycell(x, y, utilitys_temp)
                        if (utilitys_temp[x][y] - utilitys[x][y]) > delta:
                            delta = (utilitys_temp[x][y] - utilitys[x][y])
            self.deltas.append(delta)
            if delta < self.epsilon * (1 - self.discount_factor) / self.discount_factor: # taken from Stuart, Russell, and Norvig Peter. "Artificial intelligence-a modern approach 3rd ed." (2016)
                self.utilitys = utilitys_temp.copy()
                logging.info(f"Converged value iteration after {i} iterations")
                self.iterations = i
                return 

    def solve(self):
       self.valueiteration()
       self.findpath()
       self.steps = len(self.path)
       self.time = datetime.now() - self.start_time
       self.time = self.time.total_seconds()
       logging.info(f"Solved maze in {self.time} seconds and {self.steps} steps.")

class MDPPolicyIteration(MDP):
    def __init__(self, mazemanager, discount_factor=0.99, rs= 0, epsilon=0.3, debug=False):
        super().__init__(mazemanager, discount_factor, rs)
        self.initializepolicy()
        self.epsilon = epsilon
        self.debug = debug
        self.indexes =  self.get_state_index()
    def initializepolicy(self):
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                # check if wall
                if self.grid[x][y] == 1:
                    self.policy[x][y] = None
                # check if exit
                elif (x, y) == self.exit:
                    self.policy[x][y] = (0, 0)
                else:
                    neighbours = self.mazemanager.getNeigbours(x, y)
                    # take random neighbour
                    neighbour = neighbours[np.random.randint(0, len(neighbours))]
                    direction_x = int(neighbour[0] - x)
                    direction_y = int(neighbour[1] - y)
                    direction = np.array([direction_x, direction_y]) # direction to take 
                    self.policy[x][y] = direction
    
    def policyiteration(self):
        i = 0
        while True:
            self.utilitys = self.policyevaluation(self.policy)
            unchanged = True
            i += 1
            for x in range(self.grid.shape[0]):
                for y in range(self.grid.shape[1]):
                    if np.isnan(self.values[x][y]) == False:
                        uttility,  direction = self.calculateutilitycell(x, y, self.utilitys)
                        if (x, y) == self.exit:
                            continue
                        if np.array_equal(direction, self.policy[x][y]) == False:
                            self.policy[x][y] = direction
                            unchanged = False
            if self.debug:
                self.plot_policy(filename=f"Debugg/policy_{i}.png")
            if unchanged == True:
                logging.info(f"Converged policy iteration after {i} iterations")
                self.iterations = i
                return 
           
    def solve(self):
        self.start_time = datetime.now()
        self.policyiteration()
        self.findpath()
        self.steps = len(self.path)
        self.time = datetime.now() - self.start_time
        self.time = self.time.total_seconds()
        logging.info(f"Solved maze in {self.time} seconds and {self.steps} steps.")

    def get_state_index(self):
        indexes = []
        for j in range(self.grid.shape[0]):
            for i in range(self.grid.shape[1]):
                # check if exit
                if (j, i) == self.exit:
                    continue
                if np.isnan(self.values[j][i]) == False:
                    indexes.append((j, i))

        return indexes

    def policyevaluation(self,policy):
        utilitys_temp = np.zeros(self.grid.shape)
        utilitys_temp[self.exit] = 1
        while True: 
            delta = 0
            for index in self.indexes:
                    x = index[0]
                    y = index[1]
                    utility = 0
                    nextcell_x = int(x + policy[x][y][0])
                    nextcell_y = int(y + policy[x][y][1])
                    nextcell = (nextcell_x, nextcell_y)
                    utility = utilitys_temp[nextcell]
                    utility += self.values[x][y]
                    utility *= self.discount_factor
                    delta = max(delta, abs(utility - utilitys_temp[x][y]))
                    utilitys_temp[x][y] = utility
            if delta < self.epsilon:
                return utilitys_temp