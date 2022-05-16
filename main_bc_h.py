from re import X
from tkinter import Y
import turtle as t
from numpy import cumsum

class Graph:
    def __init__(self):
        self.edges = {}
        self.weights = {}

    def neighbors(self, id):
        return self.edges[id]

    def cost(self, from_node, to_node):
        return self.weights[(from_node, to_node)]


def heuristic(H ,id):
    """
    Builds Romania heuristic
    """
    L1 = {
        '0,0': 1, '0,1': 1, '0,2': 1, '0,3': 1, '0,4': 1,
        '1,0': 1, '1,1': 1, '1,2': 1, '1,3': 1, '1,4': 1,
        '2,0': 1, '2,1': 1, '2,2': 1, '2,3': 1, '2,4': 1,
        '3,0': 1, '3,1': 1, '3,2': 1, '3,3': 1, '3,4': 1,
        '4,0': 1, '4,1': 1, '4,2': 1, '4,3': 1, '4,4': 1,
        '5,0': 1, '5,1': 1, '5,2': 1, '5,3': 1, '5,4': 1,
        '6,0': 1, '6,1': 1, '6,2': 1, '6,3': 1, '6,4': 1,
        '7,0': 1, '7,1': 1, '7,2': 1, '7,3': 1, '7,4': 1,
        '8,0': 1, '8,1': 1, '8,2': 1, '8,3': 1, '8,4': 1,
        '9,0': 1, '9,1': 1, '9,2': 1, '9,3': 1, '9,4': 1
    }
    
    L2 = {
        '0,0': 1, '0,1': 1, '0,2': 1, '0,3': 1, '0,4': 1,
        '1,0': 1, '1,1': 1, '1,2': 1, '1,3': 1, '1,4': 1,
        '2,0': 1, '2,1': 1, '2,2': 1, '2,3': 1, '2,4': 1,
        '3,0': 1, '3,1': 1, '3,2': 1, '3,3': 1, '3,4': 1,
        '4,0': 1, '4,1': 1, '4,2': 1, '4,3': 1, '4,4': 1
    }

    return L1[id] if H else L2[id]

# TODO: MODIFY POSITIONS / CHECK THE IMG
def geo_pos(G, id):
    """
    Builds Romania's cities positional information 
    The map is a png image used as backgroud, 
    the position corresponds to an approximated pixel
    for each city
    """
    G1 = {
        '0,0': (1, 1), '0,1': (1, 1), '0,2': (1, 1), '0,3': (1, 1), '0,4': (1, 1),
        '1,0': (1, 1), '1,1': (1, 1), '1,2': (1, 1), '1,3': (1, 1), '1,4': (1, 1),
        '2,0': (1, 1), '2,1': (1, 1), '2,2': (1, 1), '2,3': (1, 1), '2,4': (1, 1),
        '3,0': (1, 1), '3,1': (1, 1), '3,2': (1, 1), '3,3': (1, 1), '3,4': (1, 1),
        '4,0': (1, 1), '4,1': (1, 1), '4,2': (1, 1), '4,3': (1, 1), '4,4': (1, 1),
        '5,0': (1, 1), '5,1': (1, 1), '5,2': (1, 1), '5,3': (1, 1), '5,4': (1, 1),
        '6,0': (1, 1), '6,1': (1, 1), '6,2': (1, 1), '6,3': (1, 1), '6,4': (1, 1),
        '7,0': (1, 1), '7,1': (1, 1), '7,2': (1, 1), '7,3': (1, 1), '7,4': (1, 1),
        '8,0': (1, 1), '8,1': (1, 1), '8,2': (1, 1), '8,3': (1, 1), '8,4': (1, 1),
        '9,0': (1, 1), '9,1': (1, 1), '9,2': (1, 1), '9,3': (1, 1), '9,4': (1, 1)
    }

    G2 = {
        '0,0': (1, 1), '0,1': (1, 1), '0,2': (1, 1), '0,3': (1, 1), '0,4': (1, 1),
        '1,0': (1, 1), '1,1': (1, 1), '1,2': (1, 1), '1,3': (1, 1), '1,4': (1, 1),
        '2,0': (1, 1), '2,1': (1, 1), '2,2': (1, 1), '2,3': (1, 1), '2,4': (1, 1),
        '3,0': (1, 1), '3,1': (1, 1), '3,2': (1, 1), '3,3': (1, 1), '3,4': (1, 1),
        '4,0': (1, 1), '4,1': (1, 1), '4,2': (1, 1), '4,3': (1, 1), '4,4': (1, 1)
    }

    return G1[id] if G else G2[id]


def draw_square(graph, node_id, color="medium sea green", scale=1, ts=None, text=None):
    """
    Draw a square in turtle over the map background to 
    indicate the exapnsion of the search and the shortest path
    node_id: the corresponding node in the graph (city)
    correction: corrects the origin of the geo_pos pixels
    ts: a turtle object can be passed to the function, 
    if not a turtle object is created
    To animante the shortest path a single turtle object is 
    define outside the function and passed as parameter
    """
    
    # correctionx = 600 if graph else 182
    correctiony = 327 if graph else 196
    
    if ts == None:
        ts = t.Turtle(shape="square")
    ts.shapesize(0.5, 0.5)
    ts.color(color)
    ts.penup()
    x, y = geo_pos(node_id)
    ts.goto(x*scale, correctiony - y*scale)
    if text != None:
        ts.write(str(text), font=("Arial", 20, "normal"))


def pathfromOrigin(origin, n, parents):
    # Builds shortest path from search result (parents)
    if origin == n:
        return []

    pathO = [n]
    i = n

    while True:
        i = parents[i]
        pathO.insert(0, i)
        if i == origin:
            return pathO


def costOfPath(path, graph):
    # Returns the cumulated cost of path
    cum_costs = [0]
    for i in range(len(path)-1):
        cum_costs += [graph.cost(path[i], path[i+1])]

    return cumsum(cum_costs)


def getF(oL):
    """
    Returns costs of queue F = C + H
    C: cost of route
    H: heuristic cost 
    """
    return [i[1] for i in oL]


def findleastF(oL):
    """
    finds the node with least F in oL (queue)
    This is equivalent to build a priority queue
    """
    mF = min(getF(oL))
    for ni in range(len(oL)):
        if oL[ni][1] == mF:
            return oL.pop(ni)[0]


def aStar(wgraph, graph, start, target):
    openL = []
    openL.append((start, 0))
    parents = {}
    costSoFar = {}
    parents[start] = None
    costSoFar[start] = 0

    while bool(len(openL)):
        current = findleastF(openL)
        draw_square(wgraph, current)  # Draw search expansion

        if current == target:
            break

        for successor in graph.neighbors(current):
            newCost = costSoFar[current] + graph.cost(current, successor)
            if successor not in costSoFar or newCost < costSoFar[successor]:
                costSoFar[successor] = newCost
                priority = newCost + heuristic(wgraph, successor)
                openL.append((successor, priority))
                parents[successor] = current

        print(openL)

    return parents


def main(argv):
    
    """
    Usage:
      python main_bc_h.py graph startNode
      Target for graph 0 will always be (4,0)
      Target for graph 1 will always be (9,4)
      <0> <startNode>:  Any coord from (0,0) to (4,0)
      <1> <startNode>:  Any coord from (0,0) to (9,4)
      
    Example:
      python main_bc_h.py 0 0,0
      python main_bc_h.py 1 0,0
    """
    
    if len(argv) != 2:
        print(main.__doc__)
    else:
        # print(argv[0], argv[1])
        graph = argv[0]
        startNode = argv[1]

        # Always Bucarest due to heuristic is given for this end city
        endGraph1 = '9,4'
        endGraph2 = '4,0'

        G1 = Graph()  # Builds a larger maze Graph
        G2 = Graph()  # Builds a shorter maze Graph

        # Adding edges (adjacency list)
        # TODO: Fix graph
        G1.edges = {
            'Arad': ['Zerind', 'Timisoara', 'Sibiu'],
            'Zerind': ['Arad', 'Oradea'],
            'Timisoara': ['Arad', 'Lugoj'],
            'Oradea': ['Zerind', 'Sibiu'],
            'Lugoj': ['Timisoara', 'Mehadia'],
            'Mehadia': ['Lugoj', 'Dobreta'],
            'Dobreta': ['Mehadia', 'Craiova'],
            'Sibiu': ['Arad', 'Oradea', 'Rimnicu', 'Fagaras'],
            'Rimnicu': ['Sibiu', 'Craiova', 'Pitesi'],
            'Craiova': ['Dobreta', 'Rimnicu', 'Pitesi'],
            'Fagaras': ['Sibiu', 'Bucarest'],
            'Pitesi': ['Rimnicu', 'Craiova', 'Bucarest'],
            'Bucarest': ['Fagaras', 'Pitesi']
        }
        
        G2.edges = {
            '0,4': ['0,3'],
            '0,3': ['0,2', '0,4'],
            '0,2': ['0,1', '0,3'],
            '0,1': ['1,1', '0,2'],
            '1,1': ['2,1', '0,1'],
            '2,1': ['3,1', '1,1'],
            '3,1': ['3,0', '3,2'],
            '3,0': ['2,0', '3,1'],
            '2,0': ['1,0', '3,0'],
            '1,0': ['0,0', '2,0'],
            '0,0': ['1,0'],
            '3,2': ['3,3', '3,1'],
            '3,3': ['4,3', '3,4'],
            '4,3': ['4,2', '3,3'],
            '4,2': ['4,1', '4,3'],
            '4,1': ['4,0', '4,2'],
            '4,0': ['4,1'],
            '3,4': ['4,4', '2,4'],
            '4,4': ['3,4'],
            '2,4': ['2,3', '3,4'],
            '2,3': ['2,2', '2,4'],
            '2,2': ['1,2', '2,3'],
            '1,2': ['1,3', '2,2'],
            '1,3': ['1,4', '1,2'],
            '1,4': ['1,3']
        }

        # Adding weights to edges
        # TODO: Fix graph
        G1.weights = {
            ('Arad', 'Zerind'): 75, ('Arad', 'Timisoara'): 118, ('Arad', 'Sibiu'): 140,
            ('Zerind', 'Arad'): 75, ('Zerind', 'Oradea'): 71,
            ('Timisoara', 'Arad'): 118, ('Timisoara', 'Lugoj'): 111,
            ('Oradea', 'Zerind'): 71, ('Oradea', 'Sibiu'): 151,
            ('Lugoj', 'Timisoara'): 111, ('Lugoj', 'Mehadia'): 70,
            ('Mehadia', 'Lugoj'): 70, ('Mehadia', 'Dobreta'): 75,
            ('Dobreta', 'Mehadia'): 75, ('Dobreta', 'Craiova'): 120,
            ('Sibiu', 'Arad'): 140, ('Sibiu', 'Oradea'): 151, ('Sibiu', 'Rimnicu'): 80, ('Sibiu', 'Fagaras'): 99,
            ('Rimnicu', 'Sibiu'): 80, ('Rimnicu', 'Craiova'): 146, ('Rimnicu', 'Pitesi'): 97,
            ('Craiova', 'Dobreta'): 120, ('Craiova', 'Rimnicu'): 146, ('Craiova', 'Pitesi'): 138,
            ('Fagaras', 'Sibiu'): 99, ('Fagaras', 'Bucarest'): 211,
            ('Pitesi', 'Rimnicu'): 97, ('Pitesi', 'Craiova'): 138, ('Pitesi', 'Bucarest'): 101,
            ('Bucarest', 'Fagaras'): 211, ('Bucarest', 'Pitesi'): 101
        }
        
        G2.weights = {
            ('0,4', '0,3'): 1,
            ('0,3', '0,2'): 1, ('0,3', '0,4'): 1,
            ('0,2', '0,1'): 1, ('0,2', '0,3'): 1,
            ('0,1', '1,1'): 1, ('0,1', '0,2'): 1,
            ('1,1', '2,1'): 1, ('1,1', '0,1'): 1,
            ('2,1', '3,1'): 1, ('2,1', '1,1'): 1,
            ('3,1', '3,2'): 1, ('3,1', '3,2'): 1,
            ('3,0', '2,0'): 1, ('3,0', '3,1'): 1,
            ('2,0', '1,0'): 1, ('2,0', '3,0'): 1,
            ('1,0', '0,0'): 1, ('1,0', '2,0'): 1,
            ('0,0', '1,0'): 1,
            ('3,2', '3,3'): 1, ('3,2', '3,1'): 1,
            ('3,3', '4,3'): 1, ('3,3', '3,4'): 1,
            ('4,3', '4,2'): 1, ('4,3', '3,3'): 1,
            ('4,2', '4,1'): 1, ('4,2', '4,3'): 1,
            ('4,1', '4,0'): 1, ('4,1', '4,2'): 1,
            ('4,0', '4,1'): 1,
            ('3,4', '4,4'): 1, ('3,4', '2,4'): 1,
            ('4,4', '3,4'): 1,
            ('2,4', '2,3'): 1, ('2,4', '3,4'): 1,
            ('2,3', '2,2'): 1, ('2,3', '2,4'): 1,
            ('2,2', '1,2'): 1, ('2,2', '2,3'): 1,
            ('1,2', '1,3'): 1, ('1,2', '2,2'): 1,
            ('1,3', '1,4'): 1, ('1,3', '1,2'): 1,
            ('1,4', '1,3'): 1
        }
        
        if graph:
            if argv[1] not in G1.edges.keys():
                return print("Coordenada no existe")
        else:
            if argv[1] not in G2.edges.keys():
                return print("Coordenada no existe")

        # Define screen and world coordinates
        screen = t.Screen()
        
        screen.setup(700, 700) 
        t.setworldcoordinates(0, 0, 700, 700) 

        # Use image as backgroud (image is 500x500 pixels)
        bg = './img/base_laberinto_1.png' if graph else './img/base_laberinto_2.png'
        t.bgpic(bg)

        # Get image anchored to left-bottom corner (sw: southwest)
        canvas = screen.getcanvas()
        canvas.itemconfig(screen._bgpic, anchor="sw")

        # Building aStar path of parents
        parents = aStar(graph, G1, startNode, endGraph1) if graph else aStar(graph, G2, startNode, endGraph2)

        # Calculating and printing the shortest path
        shortest_path = pathfromOrigin(startNode, endGraph1, parents) if graph else pathfromOrigin(startNode, endGraph2, parents)
        
        print(shortest_path)

        # Calculating the cost of the shortest path
        cost_tsp = costOfPath(shortest_path, G1) if graph else costOfPath(shortest_path, G2)

        # Draw shortest path 
        for ni in shortest_path:
            draw_square(ni, color="salmon")

        # Animate shortest path agent and include cost
        tsp = t.Turtle(shape="square")

        for i, ni in enumerate(shortest_path):
            draw_square(ni, color="dodger blue", ts=tsp, text=cost_tsp[i])

        t.exitonclick()  # Al hacer clic sobre la ventana grafica se cerrara


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
