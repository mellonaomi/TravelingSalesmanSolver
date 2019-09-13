import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

def plotMap(df):
	plt.scatter(df["X"],df["Y"])
	plt.show()

def plotPath(df,solution):
	#Order by "path"
	df = df.set_index('Node')
	df = df.loc[path]
	df.append(df.iloc[1,:])

	for i in range(0, len(path), 1):
		connect = df.iloc[i:i+2,:]
		plt.plot(df[i:i+2]["X"], df[i:i+2]["Y"], 'ro-')

	plt.show()

def getRandomPath(nodes):
	nodes = nodes.tolist()
	return rnd.sample(nodes,len(nodes))

def computeDistance(df,solution):
	#Order by "path"
	df = df.set_index('Node')
	df = df.loc[path]
	df.append(df.iloc[1,:])
	distance = 0
	for i in range(0, len(path), 1):
		connect = df.iloc[i:i+2,:]
		distance += #Compute the Eucledian Distance for the two points that are connected, accumulate in "distance" and return
	print("Distance for this path is: "+str(distance))
	return distance


df = pd.read_csv("data/china.txt")

fractionToSample = 0.001

df = df.sample(frac =fractionToSample)
print("Size of DF: "+ str(df.shape))
path = getRandomPath(df["Node"])

#plotPath(df,path)
computeDistance(df,path)