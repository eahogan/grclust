'''
Created on Nov 7, 2015

@author: hoga886
'''

import math
import numpy


def inv_sim(x):
    """ Given a distance x computes the similarity by taking 1/x.
    If x=0 then 1/x is defined to be infinity, if x=infinity then 
    1/x is defined to be 0.
    """
    if x == 0.0:
        return float("inf")
    elif x==float("inf"):
        return 0.0
    else:
        return 1/x

def exp_sim(x):
    """ Given a distance x computes the similarity by taking e^(-x^2/2).
    """
    return math.exp(-x**2/2)

def euc_dist(x,y):
    """ Euclidean distance between vectors x and y is defined to be
    \sqrt(\sum_{i=1}^len(x) (x_i-y_i)^2)
    """
    if len(x) != len(y):
        raise SyntaxError('Vectors must all be the same length')
    
    return math.sqrt(sum([(x[i] - y[i])**2 for i in range(len(x))]))

def cos_dist(x, y):
    """ Cosine similarity betweem vectors x and y is defined to be the 
    cosine of the angle between the two vectors. We can calculate the similarity 
    by taking x\dot y/(||x||*||y||). We turn this into a distance by taking
    1-similarity because the cosine similarity is bounded above by 1.
    """
    if len(x) != len(y):
        raise SyntaxError('Vectors must all be the same length')
    
    xnorm = euc_norm(x)
    ynorm = euc_norm(y)
    return 1-numpy.dot(x, y)/(xnorm*ynorm)

def ang_dist(x,y):
    """ Angular distance between two vectors x and y is the angle between 
    the two vectors divided by pi. We calculate this by using the cosine
    similarity.
    """
    if len(x) != len(y):
        raise SyntaxError('Vectors must all be the same length')
    
    cos_sim = 1-cos_dist(x,y);
    return math.acos(cos_sim)/math.pi;
    
def euc_norm(x):
    """ The Euclidean norm of a vector x is defined to be
    \sqrt(\sum_{i=1}^len(x) x_i^2)
    """
    return math.sqrt(sum([xi**2 for xi in x]))  

class Similarities:
    """ This class is used to calculate similarities between all pairs
    of points.
    """
    
    def __init__(self, points, dist=euc_dist, sim=exp_sim):
        self.points = points
        self.dist = dist
        self.sim = sim
        
        self.calc_distances()
        self.calc_similarities()
        
    def calc_distances(self):
        """ Calculates the distances between all pairs of points, stores it
        in self.dists. """
        self.dists = {}
        for name1 in self.points.keys():
            for name2 in self.points.keys():
                key = (name1, name2)
                self.dists[key] = self.dist(self.points[name1], self.points[name2])
        
    def calc_similarities(self):
        """ Calculates the similarities between all pairs of points based on 
        the distances, stores it in self.sims. """
        self.sims = {}
        for key in self.dists.keys():
            self.sims[key] = self.sim(self.dists[key])
                
class NNGraph:
    """ This class is used to create nearest neighbor graphs from a set 
    of similarities
    """
    
    def __init__(self, sims, k=5, epsilon=0.5, type='enn'):
        self.all_sims = sims
        self.type = type
        if type == 'enn':
            self.param = epsilon
        elif type == 'knn':
            self.param = k
    
    def make_graph(self):
        """ Make a nearest neighbor graph. """
        if self.type == 'enn':
            self.make_e_graph()
        elif self.type == 'knn':
            self.make_k_graph()
    
    def make_e_graph(self):
        """ Make an epsilon nearest neighbor graph, stores it in self.graph. """
        self.graph = {}
        for key in self.all_sims.keys():
            if self.all_sims[key] >= self.param:
                self.graph[key] = self.all_sims[key]
            else:
                self.graph[key] = 0.0
        
    def make_k_graph(self):
        """ Make a k nearest neighbor graph, stores it in self.graph. """
        self.graph = {}
        names = self.find_names()
        knn = {}
        for name in names:
            knn[name] = self.find_knn(name)
            #print(name, knn[name])
        
        for name in knn.keys():
            for other_name in knn.keys():
                if other_name in knn[name]:
                    self.graph[(name, other_name)] = self.all_sims[(name, other_name)]
                else:
                    self.graph[(name, other_name)] = 0.0
        
        for key in self.graph.keys():
            if not self.graph[key] == 0.0:
                self.graph[(key[1], key[0])] = self.graph[key]
                    
        
        
    def find_knn(self, name):
        """ Given a name, find the k nearest other names based on all_sims. """
        dists = []
        for key in self.all_sims.keys():
            if key[0] == name:
                dists.append(self.all_sims[key])
        
        dists.sort()
        #print(dists)
        cutoff = dists[int(len(dists)-self.param)]
        #print(cutoff)
        closest = []
        for key in self.all_sims.keys():
            if key[0] == name and self.all_sims[key] >= cutoff:
                closest.append(key[1])
                
        return closest
                
    def find_names(self):
        """ Gathers all the names seen in the key set of all_sims. """
        names = []
        for key in self.all_sims.keys():
            if not (key[0] in names):
                names.append(key[0])
            
        return names
        
        
        
                
                