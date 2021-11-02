import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
from utils import *


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
        # new distance cost
        self.dc = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):

    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    global checked_positions
    checked_positions = []
    open_list = []
    closed_list = []
    # Check if start or end node is on the obstacle
    if maze[start[0]][start[1]] == 1:
        print("Start node is not walkable terrain")
    if maze[end[0]][end[1]] == 1:
        print("End node is not walkable terrain")
    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node
        # Refresh the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal node
        if current_node == end_node:
            path = []
            current = current_node
            # accumulate parents nodes to draw the path
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position (8 neighborhoods)
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Avoid infinite loop by checking closed list
            if Node(current_node, node_position) in closed_list:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)
            checked_positions.append(new_node.position)
        # Loop through children
        for child in children:
            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    break
            else:
                # Create the f, g, and h values
                child.g = (current_node.g + np.sqrt((child.position[0]-current_node.position[0])**2+(child.position[1]-current_node.position[1])**2))
                #child.g = current_node.g + 1
                child.h = np.sqrt(((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2))
                # New cost 'distance cost' as dc
                # The weight of the distance cost has been set to make the path at least 3 grid away from the obstacles.
                child.dc = 5*distcost(child.position[0], child.position[1])
                child.f = child.g + child.h + child.dc

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g >= open_node.g:
                    break
            else:
                # Add the child to the open list
                open_list.append(child)


def pathplanning(start, end, image_path, verbose=False):
    # Running Time Check
    starttime = time.time()

    # Convert map image to binary list
    img = cv2.imread(image_path)
    #maze = img2binList(img, lenWidth=3580, GRID_SIZE=20, verbose=0) #cm, 1000 for E5-223 lobby 3580
    maze = img2binList(img, lenWidth=100, GRID_SIZE=1, verbose=0) # for test4.png
    # Start and End point setting

    print("Start =", start, '\n', "End =", end)

    # Procedure Checking
    print(" ", "Path planning Proceeding...", " ")

    path = astar(maze, start, end)
    print("Path planning Succeed")
    print("time :", time.time() - starttime)

    if verbose:
        # Print generated Path (in grid scale and meter scale)
        print("Path : ", path)
        print("Meter scale Path : ", convert2meter(path))

        # Visualizing binary map and generated path
        showmaze = np.array(maze).astype(np.uint8)
        showmaze *= 255
        showmaze = np.stack((showmaze,)*3, axis=-1)
        num_of_searched_node = 0
        """
        for walkable in walkable_plane_list(100, 100):          # checking walkable plane
        showmaze[walkable[0]][walkable[1]] = 60
        """
        for searched in checked_positions:
            showmaze[searched[0]][searched[1]] = [40, 40, 40]
        for colorpath in path:
            showmaze[colorpath[0]][colorpath[1]] = [200, 50, 200]
            num_of_searched_node += 1
        print(num_of_searched_node)

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                showmaze[start[0] - i][start[1] - j] = [0, 254, 0]
                showmaze[end[0] - i][end[1] - j] = [0, 0, 254]
        showmaze = cv2.resize(showmaze, None, fx=7, fy=7, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('A* algorithm run with distance cost', showmaze)
        cv2.waitKey(0)
        plt.imshow(DISTANCECOSTMAP, interpolation='None')
        plt.colorbar()
        plt.title('DISTANCECOSTMAP')
        plt.show()
        plt.close()  # press 'Q' to exit

    return path

if __name__ == '__main__':
    #start = (100, 55)
    #end = (30, 144) # (45,33) green sofa (87,76) desk (70, 115) tree (75, 160) dosirak (100,144) gs
    start = (10, 10)
    end = (60, 75)
    pathplanning(start, end, image_path="test4.png", verbose=True)
