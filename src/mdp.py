import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)

class MDP:
    def __init__(self, mazemanager, discount_factor=0.9, rs= 0, propabilities = [0.8, 0.1, 0.1],debug=False):
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
        self.propabilities = propabilities
        self.policy = np.zeros(shape = (self.grid.shape[0], self.grid.shape[1], 2))
        self.start_time = datetime.now()
        self.iterations = 0
        self.debug = debug

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
        try:
            neighbour_index = neighbours[0]
        except:
            self.plot_policy()
        for i, neighbour in enumerate(neighbours):
            if utilitys[neighbour[0]][neighbour[1]] > utility:
                utility_ = utilitys[neighbour[0]][neighbour[1]] * self.propabilities[0]
                # take random neighbour not the current one
                # check if there is only one neighbour
                if len(neighbours) == 1:
                    random_n = neighbours[0]
                elif i == 0:
                    random_n = neighbours[1]
                else:
                    random_n = neighbours[0]
                utility_ += utilitys[random_n[0]][random_n[1]] * self.propabilities[1]
                # utility to stay in the same cell
                utility_ += utilitys[x][y] * self.propabilities[2]
                if utility_ > utility:
                    utility = utility_
                    neighbour_index = neighbours[i]
        direction_x = int(neighbour_index[0] - x)
        direction_y = int(neighbour_index[1] - y)
        direction = np.array([direction_x, direction_y])
        utility += self.values[x][y]
        utility *= self.discount_factor 
        return utility, direction
    

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
        plt.xticks([]), plt.yticks([])
        plt.axis('off')
        # add the vector plot of the policy
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] != 1:
                    current = (i, j)
                    if np.isnan(self.policy[current[0]][current[1]][0]):
                        continue
                    plt.arrow(current[1], current[0], self.policy[current[0]][current[1]][1]*0.2, self.policy[current[0]][current[1]][0]*0.2, head_width=0.1, head_length=0.1, fc='red', ec='red', color = 'red')
        # add the utility values
        for i in range(self.utilitys.shape[0]):
            for j in range(self.utilitys.shape[1]):
                if self.grid[i][j] != 1:
                    current = (i, j)
                    plt.text(current[1], (current[0]+ 0.3), round(self.utilitys[current[0]][current[1]], 2), horizontalalignment='center', verticalalignment='center', color = 'black', fontsize = 12)
        plt.tight_layout()
        # color the cells based on the utility value
        # get the values of the utilitys
        values = self.utilitys.copy()
        # replace 0 with nan
        values[values == 0] = np.nan
        # change plot size
        colors = plt.cm.viridis(values)
        # remove ticks
        plt.xticks([]), plt.yticks([])
        plt.axis('off')
        plt.tight_layout()
       # add zje colors to the grid
        plt.imshow(colors, alpha=0.5)

        plt.legend(loc = "upper left", prop={'size': 20})

        # a
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
    def __init__(self, mazemanager, discount_factor=0.9, epsilon=0.000000001, rs= 0, debug = False, propabilities = [0.8, 0.1, 0.1]):
        super().__init__(mazemanager, discount_factor, rs, propabilities)
        self.epsilon = epsilon
        self.deltas = []
        self.debug = debug
        self.indexes =  self.get_state_index()  

    def valueiteration(self):
        utilitys_temp = np.zeros(self.grid.shape)
        # set utility of exit to 1
        utilitys_temp[self.exit[0]][self.exit[1]] = 1
        i = 0
        while True:
            i += 1
            utilitys = np.copy(utilitys_temp)
            delta = 0
            for index in self.indexes:
                x = index[0]
                y = index[1]
                utilitys_temp[x][y], self.policy[x][y]  = self.calculateutilitycell(x, y, utilitys_temp)
                if (utilitys_temp[x][y] - utilitys[x][y]) > delta:
                    delta = (utilitys_temp[x][y] - utilitys[x][y])
            self.deltas.append(delta)
            if self.debug:
                self.utilitys = utilitys_temp.copy()
                self.plot_policy(f"Debugg/Value/policy_mdp_{i}.png")

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
    def __init__(self, mazemanager, discount_factor=0.9, rs= 0,  epsilon=0.000001, debug=False, propabilities = [0.8, 0.1, 0.1]):
        super().__init__(mazemanager, discount_factor, rs, propabilities)
        self.initializepolicy()
        self.indexes =  self.get_state_index()
        self.epsilon = epsilon
        self.debug = debug
        
       
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
            for index in self.indexes:
                x = index[0]
                y = index[1]
                utility,  direction = self.calculateutilitycell(x, y, self.utilitys)
                if np.array_equal(direction, self.policy[x][y]) == False:
                    unchanged = False
                self.policy[x][y] = direction
            if self.debug:
                self.plot_policy(filename=f"Debugg/Policy/policy_mdp_{i}.png")
            # print(f"iteration {i}")
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

    def policyevaluation(self,policy):
        utilitys_temp = self.utilitys.copy()
        # utilitys_temp = np.zeros(self.grid.shape)
        utilitys_temp[self.exit] = 1
        i = 0
        while True:
            delta = 0
            i += 1
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
                    delta = max(delta, np.abs(utility - utilitys_temp[x][y]))
                    utilitys_temp[x][y] = utility
            logging.debug(f"Policy evaluation iteration {i} delta: {delta}")
            if delta < self.epsilon * (1 - self.discount_factor) / self.discount_factor:
                logging.debug(f"Converged policy evaluation after {i} iterations")
                self.utilitys = utilitys_temp.copy()
                return self.utilitys