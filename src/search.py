from datetime import datetime
import logging
import numpy as np
from queue import PriorityQueue
# initialize logging info level
logging.basicConfig(level=logging.INFO)
class Search:
    def __init__(self, mazemanager):
        self.mazemanager = mazemanager
        self.start = mazemanager.entrance
        self.end = mazemanager.exit
        self.visited = []
        self.path = []
        self.queue = []
        self.queue.append(self.start)
        self.current = self.start
        self.steps = 0
    

    def reversePath(self):
        # reverse the path to get the optimal path
        current = self.end
        path = []
        while current != self.start:
            for node in self.path:
                if node[1] == current:
                    path.append(node)
                    current = node[0]
                    break
        self.path = path
        


    def solve(self):
        start = datetime.now()
        if self.current == self.end:
            return True
        # exact replica from the pseudocode from Stuart, Russell, and Norvig Peter. "Artificial intelligence-a modern approach 3rd ed." (2016)
        while True:
            if self.queue == []:
                return False
            self.current = self.getNext()
            if self.current == self.end:
                end = datetime.now()
                self.time = end - start
                # time to seconds
                self.time = self.time.total_seconds()
                logging.info(f"Solved maze in {self.time} seconds and {self.steps} steps.")
                return True
            self.visited.append(self.current)
            for neighbour in self.mazemanager.getNeigbours(self.current[0], self.current[1]):
                if neighbour not in self.visited and neighbour not in self.queue:
                    self.addtoqueue(neighbour)
                    self.path.append((self.current, neighbour))
                    self.steps += 1


    def getNext(self): # abstract method to be implemented by the subclasses
        pass

    def addtoqueue(self, node):
        self.queue.append(node)

class Breadthfirst (Search):
    def __init__(self, mazemanager):
        super().__init__(mazemanager)

    def getNext(self):
        return self.queue.pop(0) # this is a FIFO queue
       
      

class Depthfirst (Search):
    def __init__(self, mazemanager):
        super().__init__(mazemanager)

    def getNext(self):
        return self.queue.pop() # this is a LIFO queue

class Astar (Search):
    def __init__(self, mazemanager, heuristic = "manhattan"):
        super().__init__(mazemanager)
        self.start = mazemanager.entrance
        self.end = mazemanager.exit
        self.path = []
        # self.visited[self.start] = 0
        self.cost = dict()
        self.cost[self.start] = 0
        self.current = self.start
        self.pqueue = PriorityQueue()
        self.pqueue.put((0, self.start))
        self.steps = 0

        if heuristic == "manhattan":
            self.heuristic = self.manhattan
        elif heuristic == "euclidian":
            self.heuristic = self.euclidian


    def manhattan(self, node):
        # calculate distance from node to exit with manhattan distance

        manhattan = np.sum(np.abs(np.array(node) - np.array(self.end)))

        return manhattan

    
    def euclidian(self, node):
        # calculate distance from node to exit with euclidian distance

        euclidian = np.sqrt(np.sum(np.square(np.array(node) - np.array(self.end))))

        return euclidian
    

    def getNext(self):
        _, self.current = self.pqueue.get()
        return self.current
    

    def addtoqueue(self, heuristic_value, node):
        self.pqueue.put((heuristic_value, node))

   
    def solve(self):
        start = datetime.now()
        if self.current == self.end:
            return True
        while True:  
            self.current = self.getNext()
            if self.current == self.end:
                end = datetime.now()
                self.time = end - start
                # time to seconds
                self.time = self.time.total_seconds()
                logging.info(f"Solved maze in {self.time} seconds and {self.steps} steps.")
                return True
            for neighbour in self.mazemanager.getNeigbours(self.current[0], self.current[1]):
                current_cost = self.cost[self.current] + 1 # +1 since mahattan distance is always 1 bewteen neighbor and current
                if neighbour not in self.cost.keys() or current_cost < self.cost[neighbour]:
                    self.cost[neighbour] = current_cost
                    heuristic_value = current_cost + self.heuristic(neighbour)
                    self.addtoqueue(heuristic_value, neighbour)
                    self.path.append((self.current, neighbour))
                    self.steps +=1
        
        


