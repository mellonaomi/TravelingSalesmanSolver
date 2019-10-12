import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import sys


def plotMap(df):
	plt.scatter(df["X"],df["Y"])
	plt.show()

def eucDistance(point1,point2):
	return np.linalg.norm(point1 - point2, axis=0) 

def plotPath(df,solution):
	#Order by "path"
	print(df)
	df = df.set_index('Node')
	df = df.loc[solution]
	print(df)
	df.append(df.iloc[1,:])

	for i in range(0, len(solution), 1):
		connect = df.iloc[i:i+2,:]
		#print(df.iloc[i:i+2]["X"])
		#print(df.iloc[i:i+2]["Y"])
		#print("+====+")
		plt.plot(df.iloc[i:i+2]["X"], df.iloc[i:i+2]["Y"], 'ro-')


	plt.show()

def getRandomPath(nodes):
	nodes = nodes.tolist()
	return rnd.sample(nodes,len(nodes))

def computeDistance(df,solution):
	#Order by "path"
	df = df.set_index('Node')
	df = df.loc[solution]
	df.append(df.iloc[1,:])
	distance = 0
	#Compute the Eucledian Distance for the two points that are connected, accumulate in "distance" and return
	for i in range(0, len(solution)-1, 1):
		point1 = df.iloc[i,:]
		point2 = df.iloc[i+1,:]
		#print(point1)
		#print("and")
		#print(point2)	
		distance += eucDistance(point1,point2)
	print("Distance for this path is: "+str(distance))
	return distance


def solution_Greedy(df):
	path = [df.iloc[0,]["Node"]]
	for i in range(0,len(df["Node"])):
		currentPoint = df.iloc[i,]
		#Find "closest point"
		bestDistance = sys.maxsize
		remainingPoints = df[~df["Node"].isin(path)]
		if len(path) != len(df["Node"]):
			for j in range(0,len(remainingPoints["Node"])):
				nxt = remainingPoints.iloc[j,:]
				#print(nxt)
				#print(currentPoint)
				#print(str(currentPoint["Node"]) + "   --_> " +str(nxt["Node"]))
				dst = eucDistance(currentPoint,nxt)
				if dst < bestDistance and currentPoint["Node"] != nxt["Node"]:
					#print("Best distance: " + str(nxt["Node"]) + " D: " + str(dst))
					bestDistance = dst
					candidate = nxt
			#print("Appending: " + str(candidate["Node"]))
			path.append(candidate["Node"])
	#print("Final: ")
	#print(path)
	path = [int(p) for p in path]
	return path


df = pd.read_csv("data/china.txt")

fractionToSample = 0.001

df = df.sample(frac =fractionToSample)
print("Size of DF: "+ str(df.shape))
path = getRandomPath(df["Node"])#[1:3]


df = df[df["Node"].isin(path)]
# plotPath(df,path)
#print(computeDistance(df,path))
#print(df)
testCase = pd.DataFrame({
	'Node':[0,1,2,3],
	'X':[0,90,2,0.5], 
	'Y':[0,1,5,0.5]}
	,index = [0,1,2,3])

testPath = [0,1,2,3]
#print(df2)
#computeDistance(testCase,testPath)
solution = solution_Greedy(df)
plotPath(df,solution)