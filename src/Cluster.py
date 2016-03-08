'''
Created on Nov 7, 2015

@author: hoga886
'''

import sys
import csv
import argparse
from sklearn import cluster
import numpy

import GraphMethods as gm

def read_data(file):
    """ Reads csv file and returns dictionary mapping first element 
    in each row (naming the row) to a vector of the rest of the 
    elements in that row"""
    
    points = {}
    names = []
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ",")
        for row in reader:
            name = row[0]
            names.append(name)
            point = [float(row[i]) for i in range(1,len(row))]
            points[name] = point
            
    return points, names

def main():
    parser = argparse.ArgumentParser(description="Parse clustering inputs")
    parser.add_argument('-f','--infile', type=str, required=True, dest='in_file', help="Name of input file")
    parser.add_argument('-o','--outfile', required=True, dest='out_file', help="Name of output file")
    parser.add_argument('-c', '--clustType', choices=['spectral','kmeans','fiedler'], default='fiedler', dest='cluster', help="Type of clustering")
    parser.add_argument('-k', type=int, dest='k', help="Number of clusters")
    parser.add_argument('-nn', nargs=2, dest='nn_type', help="Type of nearest neighbor graph and parameter")
    args = parser.parse_args()
    
    points, names = read_data(args.in_file)
    #print(points)
    #print(names)
   
    
    if args.cluster == "spectral":
        sims = gm.Similarities(points)
        print(sims.sims)
        nn = gm.NNGraph(sims.sims, type = args.nn_type[0], k=float(args.nn_type[1]))
        nn.make_graph()
        
        affinity_matrix = numpy.zeros((len(names), len(names)))
        for i in range(len(names)):
            for j in range(len(names)):
                affinity_matrix[i,j] = nn.graph[(names[i], names[j])]
        
        #print(affinity_matrix)
        
        spectral = cluster.SpectralClustering(n_clusters=args.k, affinity='precomputed')
        spectral.fit(affinity_matrix)
        
        labels = spectral.labels_
        #print(labels)
        
    elif args.cluster == "kmeans":
        kmeans = cluster.KMeans(args.k)
        data = []
        for i in range(len(names)):
            print(names[i])
            data.append(points[names[i]])
        kmeans.fit(data)
        labels = kmeans.labels_
        
    elif args.cluster == "fiedler":
        sims = gm.Similarities(points)
        print(sims.sims)
        nn = gm.NNGraph(sims.sims, type = args.nn_type[0], k=float(args.nn_type[1]))
        nn.make_graph()
        
        affinity_matrix = numpy.zeros((len(names), len(names)))
        for i in range(len(names)):
            for j in range(len(names)):
                affinity_matrix[i,j] = nn.graph[(names[i], names[j])]
        
        fiedler = cm.FiedlerClustering(n_clusters=args.k)
        fiedler.fit(affinity_matrix)
        
        labels = fiedler.labels_
    
    labeler = {}
    for i in range(len(names)):
        labeler[names[i]] = labels[i]

    print(labeler)

if __name__ == "__main__":
    main() 