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

def computeDistance(df,solution,verbose = False):
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
	if verbose == True:
		print("Distance for this path is: "+ str(distance))
	return distance


def solution_greedy(df):
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

def annealing_temperature(fraction):
	return max(0.01, min(1, 1 - fraction))

def annealing_acceptance_probability(cost, new_cost, temperature):
	if new_cost < cost:
		return 1
	else:
		p = np.exp(- (new_cost - cost) / temperature)
		return p

def annealing_random_step(nodes):
	#nodes = nodes.tolist()
	i = j = 0
	while i==j:
		i = rnd.sample(range(len(nodes)),1)[0]
		j = rnd.sample(range(len(nodes)),1)[0]
	#Swap
	temp = nodes[i]
	nodes[i] = nodes[j]
	nodes[j] = temp
	return nodes

def annealing_plot(states, costs):
	plt.figure()
	plt.suptitle("Cost of Simmulated Annealing")
	plt.plot(costs, 'b')
	plt.title("Costs")
	plt.show()



def solution_annealing(df,maxsteps=1000,debug=True):
	state = getRandomPath(df["Node"])
	cost = computeDistance(df,state)
	states, costs = [state], [cost]
	for step in range(maxsteps):
		fraction = step / float(maxsteps)
		T = annealing_temperature(fraction)
		new_state = annealing_random_step(state)
		new_cost = computeDistance(df,new_state)
		if debug: 
			print(str(step) + " --> C: " + str(cost) )
		if annealing_acceptance_probability(cost, new_cost, T) > rnd.random():
				state, cost = new_state, new_cost		
				states.append(state[:])
				costs.append(cost)		
	index_best = min(range(len(costs)), key=costs.__getitem__)
	return states[index_best], costs[index_best], states, costs



df = pd.read_csv("data/china.txt")



fractionToSample = 0.001


df = df.sample(frac = fractionToSample)
print("Size of DF: "+ str(df.shape))

path = getRandomPath(df["Node"])#[1:30]
#distance1 = computeDistance(df,path)



df = df[df["Node"].isin(path)]
# plotPath(df,path)
#print(computeDistance(df,path))
#print(df)
testCase = pd.DataFrame({
	'Node':[0,1,2,3,4,5,6],
	'X':[0,90,2,0.5,2,1,8], 
	'Y':[0,1,5,0.5,9,4,3]}
	,index = [0,1,2,3,4,5,6])

testPath = [0,1,2,3,4,5,6]
#print(df2)
#computeDistance(testCase,testPath)
#df = testCase
result_greedy = solution_greedy(df)
result_random = getRandomPath(df["Node"])


result_annealing, cost_annealing, states, costs = solution_annealing(df,maxsteps=1000,debug=False)
#annealing_plot(states,costs)
#plotPath(df,final_state)

print("Random Cost: " + str(computeDistance(df,result_random)))
print("Greedy Cost: " + str(computeDistance(df,result_greedy)))
print("Simmulated Annealing Cost: " + str(computeDistance(df,result_annealing)))
