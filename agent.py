# SOLVER CLASSES WHERE AGENT CODES GO
from helper import *
import random
import math


# Base class of agent (DO NOT TOUCH!)
class Agent:
    def getSolution(self, state, maxIterations):
        return []       # set of actions


#####       EXAMPLE AGENTS      #####

# Do Nothing Agent code - the laziest of the agents
class DoNothingAgent(Agent):
    def getSolution(self, state, maxIterations):
        if maxIterations == -1:     # RIP your machine if you remove this block
            return []

        #make idle action set
        nothActionSet = []
        for i in range(20):
            nothActionSet.append({"x":0,"y":0})

        return nothActionSet

# Random Agent code - completes random actions
class RandomAgent(Agent):
    def getSolution(self, state, maxIterations):

        #make random action set
        randActionSet = []
        for i in range(20):
            randActionSet.append(random.choice(directions))

        return randActionSet


#####    ASSIGNMENT 1 AGENTS    #####
# BFS Agent code
class BFSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        intializeDeadlocks(state)
        iterations = 0
        bestNode = None
        queue = [Node(state.clone(), None, None)]   #state, parent, action
        visited = []

        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            # YOUR CODE HERE
            if iterations == 1:                 # Initializing bestNode variable to 1st node only once
                bestNode = queue[0]
            currNode = queue.pop(0)             # currNode = first node of list (FIFO Queue implementation)
            if bestNode.getHeuristic() == currNode.getHeuristic():
                if bestNode.getCost() > currNode.getCost():
                    bestNode = currNode         # if heuristic is same, comparing cost and setting bestNode
            elif bestNode.getHeuristic() > currNode.getHeuristic():
                bestNode = currNode             # if heuristic is not same, setting bestNode
            if bestNode.checkWin():
                break                           # if win state reached, break out of while loop to reach return
            if currNode.getHash() in visited:
                continue                        # if currNode already in visited, don't do anything
            visited.append(currNode.getHash())  # adding getHash function of currNode to visited
            children = currNode.getChildren()
            for child in children:
                if child.getHash() not in visited:  # checking child in visited without increasing complexity
                    queue.append(child)             # if child was not in visited, we add the child to queue

        return bestNode.getActions()


# DFS Agent Code
class DFSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        intializeDeadlocks(state)
        iterations = 0
        bestNode = None
        queue = [Node(state.clone(), None, None)]
        visited = []
        #expand the tree until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            # YOUR CODE HERE
            if iterations == 1:                 # Initializing bestNode variable only once
                bestNode = queue[0]
            currNode = queue.pop()              # currNode = last node of list (LIFO Stack implementation)
            if bestNode.getHeuristic() == currNode.getHeuristic():
                if bestNode.getCost() > currNode.getCost():
                    bestNode = currNode
            elif bestNode.getHeuristic() > currNode.getHeuristic():
                bestNode = currNode
            if bestNode.checkWin():
                break
            if currNode.getHash() in visited:
                continue
            visited.append(currNode.getHash())
            children = currNode.getChildren()
            for child in children:
                if child.getHash() not in visited:
                    queue.append(child)

        return bestNode.getActions()


# AStar Agent Code
class AStarAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        #setup
        intializeDeadlocks(state)
        iterations = 0
        bestNode = None

        #initialize priority queue
        queue = PriorityQueue()
        queue.put(Node(state.clone(), None, None))
        visited = []
        while (iterations < maxIterations or maxIterations <= 0) and queue.qsize() > 0:
            iterations += 1

            ## YOUR CODE HERE ##
            if iterations == 1:
                bestNode = Node(state.clone(), None, None)  # Initialization of bestNode only once
            currNode = queue.get()              # currNode = first node of Priority Queue
            if bestNode.getHeuristic() == currNode.getHeuristic():
                if bestNode.getCost() > currNode.getCost():
                    bestNode = currNode
            elif bestNode.getHeuristic() > currNode.getHeuristic():
                bestNode = currNode
            if bestNode.checkWin():
                break
            if currNode.getHash() in visited:
                continue
            visited.append(currNode.getHash())
            for child in currNode.getChildren():
                cost = child.getHeuristic() + child.getCost()
                if child.getHash() not in visited:
                    queue.put(child, cost)

        return currNode.getActions()

class KNN:
	def __init__(self, k):
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		self.xTrain = X
		self.yTrain = y
		return 0

	def predict(self, X):
		k_Neigh = list()
		sol = list()
		for testData in X:
			n_list = list()
			for i in range (0,len(self.xTrain)):
				euc = self.distance(self.xTrain[i],testData)
				dt_tuple = (self.xTrain[i],self.yTrain[i],euc)
				n_list.append(dt_tuple)
			n_list.sort(key = lambda x: x[2])
			n_list = n_list[:self.k]
			k_Neigh.append(n_list)
		for i in k_Neigh:
			p = [n[1] for n in i]
			l = max(p, key = p.count)
			sol.append(l)
		return np.asarray(sol)


class Perceptron:
	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w
		self.b = b

	def step_function(self,x):
		return 1 * (x > 0)

	def train(self, X, y, steps):
		for p in range(steps):
			i = p % y.size
			pred = self.predict(X[i])
			self.w = self.w - ( X[i] * (y[i] - pred) * self.lr)
			self.b = self.b - ( (y[i] - pred) * self.lr)
		return None

	def predict(self, X):
		dp = np.dot(X, self.w) + self.b
		pred = self.step_function(dp)
		return pred


class ID3:
	class Tree(object):
		def __init__(self,attr):
			self.attr = attr
			self.parent = None
			self.branches = {}
		def add_Branch(self,key,branch):
			self.branches[key] = branch
			if type(branch) == type(self):
				self.branches[key].parent = self
		def str_method(self):
			s1 = " "
			if self.parent != None:
				s1 += str(self.parent.attr)+" --> "+str(self.attr)
			else:
				s1 += str(self.attr)
			for i in self.branches:
				if type(self.branches[i]) == type(self):
					s1 += "\n\tbranch: "+str(i)+"\tattribute: "+str(self.branches[i].attr)
				else:
					s1 += "\n\tbranch: "+str(i)+"\tout: "+str(self.branches[i])
			return s1

	class Node(object):

		def str_method(self):
			s1 = "ID : "+str(self.id)+"\t\tData : "+str(self.data)+"\n\t\tOutcome : "+str(self.value)
			return s1
		def __init__(self,id,data,value):
			self.id = id
			self.value = value
			self.data = data

	def __init__(self, nbins, data_range):
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		categorical_data = self.preprocess(X)
		ex = []
		attr = np.arange(np.size(categorical_data,1))
		for d in range(len(categorical_data)):
			n_ex = self.Node(d,categorical_data[d],y[d])
			ex.append(n_ex)
		self.tree = self.build_tree_func(ex,attr,None)

	def build_tree_func(self,ex,attr_s,par_ex):
		bestAttr = None
		for i in attr_s:
			totalExample = len(ex)
			val = {}
			for j in ex:
				if j.data[i] not in val:
					val[j.data[i]] = [j]
				else:
					val[j.data[i]].append(j)
			total = len(ex)
			present_total = 0
			for j in val:
				pvalue,nvalue = self.value_PN(val[j])
				present_total += pvalue/total*self.Comp_info_val(pvalue,nvalue)
			P_examples,N_examples = self.value_PN(ex)
			expect_info = self.Comp_info_val(P_examples,N_examples)
			sol = expect_info - present_total
			if bestAttr == None or sol > bestAttr[1]:
				bestAttr= (i,sol)
		is_it_same = True
		last_Val = ex[0].value
		for i in ex:
			if i.value != last_Val:
				is_it_same = False
				break
		if len(attr_s)==0:
			l = self.plurality(ex)
			return l
		elif ex == None:
			p = self.plurality(par_ex)
			return p
		elif ex == is_it_same:
			q = ex[-1].value
			return q
		else:
			target = bestAttr[0]
			n_tree = self.Tree(target)
			new_attribute = attr_s[attr_s!=target]
			totalExample = len(ex)
			val = {}
			for i in ex:
				if i.data[target] not in val:
					val[i.data[target]] = [i]
				else:
					val[i.data[target]].append(i)
			for i in val:
				n_ex = []
				for j in ex:
					if j.data[target] == i:
						n_ex.append(j)
				sub_tree = self.build_tree_func(n_ex,new_attribute,ex)
				n_tree.add_Branch(i,sub_tree)
		return n_tree

	def Comp_info_val(self,p_val,n_val):
		tot = p_val + n_val
		if tot != 0:
			n = (n_val/tot)
			p = (p_val/tot)
			if p != 0:
				sol = np.log2(p)*(-(p))
			if n != 0:
				sol = np.log2(n)*(-(n))
		return sol

	def value_PN(self,ex):
		ex_P =0
		ex_N = 0
		for i in ex:
			if i.value == 1:
				ex_P = ex_P + 1
			else:
				ex_N = ex_N + 1
		return ex_P,ex_N

	def plurality(self,ex):
		major = {}
		for i in ex:
			opt = i.value
			if opt not in major:
				major[opt] = 1
			else:
				major[opt] += 1
		result = max(major,key=lambda x:major[x])
		return result

	def display_tree(self,tree_t):
		print(tree_t)
		for branch_1 in tree_t.branches:
			if type(tree_t.branches[branch_1]) == type(tree_t):
				self.display_tree(tree_t.branches[branch_1])
		return None

	def input_data(self,tree_t,data):
		outcome = self.Get_result(tree_t)
		sol1 = 0
		sol2 = 0
		for i in outcome:
			if i == 0:
				sol1 = sol1 + 1
			else:
				sol2 = sol2 + 1
		if sol1 > sol2:
			p = 0
		elif sol2 > sol1:
			p = 1
		else:
			p = np.random.randint(0,1)
		data_v = data[tree_t.attr]
		if data_v not in tree_t.branches:
			return p
		n_tree = tree_t.branches[data_v]
		if type(n_tree) == type(tree_t):
			return self.input_data(n_tree,data)
		else:
			return n_tree

	def Get_result(self,tree_t):
		outcome = []
		for i in tree_t.branches:
			branch_m = tree_t.branches[i]
			if type(branch_m) == type(self.tree):
				res = self.Get_result(branch_m)
				outcome += res
			else:
				outcome.append(branch_m)
		return outcome


	def predict(self, X):
		categorical_data = self.preprocess(X)
		pred = np.array([])
		for i in categorical_data:
			pred = np.append(pred,self.input_data(self.tree,i))
		return pred


class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)


class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		#Write forward pass here
		return None

	def backward(self, gradients):
		#Write backward pass here
		return None


class Sigmoid:

	def __init__(self):
		None

	def forward(self, input):
		#Write forward pass here
		return None

	def backward(self, gradients):
		#Write backward pass here
		return None


class K_MEANS:

	def __init__(self, k, t):
		#k_means state here
		#Feel free to add methods
		# t is max number of iterations
		# k is the number of clusters
		self.k = k
		self.t = t

	def distance(self, centroids, datapoint):
		diffs = (centroids - datapoint)**2
		return np.sqrt(diffs.sum(axis=1))

	def train(self, X):
		#training logic here
		#input is array of features (no labels)


		return self.cluster
		#return array with cluster id corresponding to each item in dataset

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)


class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w
		self.b = b

	def forward(self, input):
		self.x = input
		self.m = np.dot(self.x,self.w)
		self.m = self.m +self.b
		return self.m

	def backward(self, gradients):
		trans_x = self.x.transpose()
		trans_w = self.w.transpose()
		w_temp = np.dot(trans_x,gradients)
		x_temp = np.dot(gradients,trans_w)
		self.w = self.w - (self.lr * w_temp)
		self.b = self.b - (self.lr * gradients)
		return x_temp


class Sigmoid:

	def __init__(self):
		None

	def forward(self, input):
		self.sig = np.array([1/(1+np.exp(-x)) for x in input])
		return self.sig

	def backward(self, gradients):
		return gradients * self.sig * (1 - self.sig)

class K_MEANS:

	def __init__(self, k, t):
		self.k = k
		self.cl = [[] for _ in range(self.k)]
		self.t = t


	def distance(self, centroids, datapoint):
		diffs = (centroids - datapoint)**2
		return np.sqrt(diffs.sum())

	def train(self, X):
		self.X = X
		Centroids = []
		old_centroids = 0
		self.nSamples, self.nFeatures = X.shape
		randomSampleIndexes = np.random.choice(self.nSamples, self.k, replace = False)
		for i in randomSampleIndexes:
			Centroids.append(self.X[i])
		for _ in range(self.t):
			old_centroids = Centroids
			clusters = [[] for _ in range(self.k)]
			for i in range(len(self.X)):
				distances = []
				for point in Centroids:
					distances.append(self.distance(self.X[i], point))
				ci = np.argmin(distances)
				clusters[ci].append(i)
			self.cl = clusters
			centroids = np.zeros((self.k, self.nFeatures))
			for i in range(0,len(self.cl)):
				clusterMean = np.mean(self.X[self.cl[i]], axis = 0)
				centroids[i] = clusterMean
			Centroids = centroids
			for i in range(self.k):
				p=0
				distances = [self.distance(old_centroids[i], Centroids[i])]
				p = sum(distances) == 0
			if (p==1):
				break
		labels = [None] * self.nSamples
		count = 0
		for z in range(0,len(self.cl)):
			for sampleIndex in self.cl[z]:
				labels[sampleIndex] = z
				count = count+1
		return labels



class AGNES:
	def __init__(self, k):
		self.k = k

	def distance(self, a, b):
		diffs = (a - b)**2
		return np.sqrt(diffs.sum())

	def train(self, X):
		clusters = []* 653
		cl_final = []
		for i in range(len(X)):
			clusters.append([i])
		dist_matrix_dup = []
		cnt = 0
		d_m = [[0]*len(X)]*len(X)
		for i in range(len(X)):
			for j in range(len(X)):
				d_m.append(0)
		for i in range(len(X)):
			for j in range(len(X)):
				if(i!=j):
					d_m[i][j] = self.distance(X[i], X[j])
				elif i==j:
					d_m[i][j] = 0

		sort_dist = sorted([(d_m[i][j], (i, j)) for i in range(len(X)) for j in range(i+1, len(X)) if i!=j], key = lambda x: x[0])

		while(sort_dist and len(clusters) != self.k):

			cnt += 1
			nextPoint = sort_dist.pop(0)
			mindistance= nextPoint[0]
			first_coordinate = nextPoint[1][0]
			second_coordinate = nextPoint[1][1]
			next_cluster_merge = None
			len_1 = len(clusters)
			for pid in range(len_1):

				cluster = clusters[pid]
				if first_coordinate in cluster or second_coordinate in cluster:
					if not next_cluster_merge:
						next_cluster_merge = pid
					else:
						if pid < next_cluster_merge:
							clusters[pid].extend(clusters[next_cluster_merge])
							ab = clusters.pop(next_cluster_merge)
						else:
							clusters[next_cluster_merge].extend(clusters[pid])
							ab = clusters.pop(pid)
						break

		for i in range(len(X)):
			for m in range(len_1):
				if i in clusters[m]:
					cnt += 1
					cl_final.append(m)
					break

		self.cluster = np.array(cl_final)
		return self.cluster


class HillClimberAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        #setup
        intializeDeadlocks(state)
        iterations = 0
        seqLen = 50            # maximum length of the sequences generated
        coinFlip = 0.5          # chance to mutate
        bestSeq = []
        for i in range(seqLen):
            bestSeq.append(random.choice(directions))
        start_node = Node(state,None,None)
        last_node = start_node
        mh = 99999999999999
        currSeq = bestSeq[:]
        while (iterations < maxIterations or maxIterations<=0):
            iterations += 1
            it = True
            for i in range(seqLen):
                if it:
                    currSeq[i] = random.choice(directions)
                    it = False
                else:
                    if(random.random()>=coinFlip):
                        currSeq[i] = random.choice(directions)
            curr_state = state.clone()
            dl = False
            for i in range(seqLen):
                curr_state.update(currSeq[i]['x'],currSeq[i]['y'])
                while(curr_state.checkWin()):
                    bestSeq = currSeq
                    return bestSeq[:i+1]
                if(checkDeadlock(curr_state)):
                    dl = True
                    break
            if dl:
                continue
            dist = getHeuristic(curr_state)
            while(dist<mh):
                mh = dist
                bestSeq = currSeq
        return bestSeq



# Genetic Algorithm code
class GeneticAgent(Agent):

    def getselectedindex(self, number):
        if number > 45 and number <= 55:
            return 0
        elif number > 36 and number <= 45:
            return 1
        elif number > 28 and number <= 36:
            return 2
        elif number > 21 and number <= 28:
            return 3
        elif number > 15 and number <= 21:
            return 4
        elif number > 10 and number <= 15:
            return 5
        elif number > 6 and number <= 10:
            return 6
        elif number > 3 and number <= 6:
            return 7
        elif number > 1 and number <= 3:
            return 8
        elif number > 0 and number <= 1:
            return 9

    def getSolution(self, state, maxIterations=-1):
        intializeDeadlocks(state)
        iterations = 0
        seqLen = 50
        popSize = 10
        parentRand = 0.5
        mutRand = 0.3
        bs = []
        population = []
        for p in range(popSize):
            bs = []
            for i in range(seqLen):
                bs.append(random.choice(directions))
            population.append(bs)
        while (iterations < maxIterations):
            iterations += 1
            cs = state.clone()
            population_sorted = []
            for p in range(popSize):
                cs = state.clone()
                for i in population[p]:
                    cs.update(i['x'], i['y'])
                win = cs.checkWin()
                heur = getHeuristic(cs)
                if (win):
                    bs = population[p]
                    return bs
                population_sorted.append((heur, population[p]))
            population_sorted.sort(key=lambda x: x[0])
            cur_hightest_h = population_sorted[0][0]
            if (cur_hightest_h < getHeuristic(state)):
                bs = population_sorted[0][1]
            new_pop = []
            q = int(popSize/2)
            for i in range(q):
                get_index1 = self.getselectedindex(random.randint(1, 55))
                get_index2 = self.getselectedindex(random.randint(1, 55))
                par1 = population_sorted[get_index1][1]
                par2 = population_sorted[get_index2][1]
                offspring = []
                if (random.random() < parentRand):
                    offspring = (par1[0:25] + par2[25:50])
                else:
                    offspring = (par2[0:25] + par1[25:50])
                for i in range(50):
                    while (random.random() < mutRand):
                        offspring[i] = random.choice(directions)
                new_pop.append(offspring)
            p = int(popSize/2)
            for i in range(p):
                new_pop.append(population_sorted[i][1])
            population = new_pop
        return bs

# MCTS Specific node to keep track of rollout and score
class MCTSNode(Node):
    def __init__(self, state, parent, action, maxDist):
        super().__init__(state,parent,action)
        self.children = []  #keep track of child nodes
        self.n = 0          #visits
        self.q = 0          #score
        self.maxDist = maxDist      #starting distance from the goal (heurstic score of initNode)

    #update get children for the MCTS
    def getChildren(self,visited):
        #if the children have already been made use them
        if(len(self.children) > 0):
            return self.children

        children = []

        #check every possible movement direction to create another child
        for d in directions:
            childState = self.state.clone()
            crateMove = childState.update(d["x"], d["y"])

            #if the node is the same spot as the parent, skip
            if childState.player["x"] == self.state.player["x"] and childState.player["y"] == self.state.player["y"]:
                continue

            #if this node causes the game to be unsolvable (i.e. putting crate in a corner), skip
            if crateMove and checkDeadlock(childState):
                continue

            #if this node has already been visited (same placement of player and crates as another seen node), skip
            if getHash(childState) in visited:
                continue

            #otherwise add the node as a child
            children.append(MCTSNode(childState, self, d, self.maxDist))

        self.children = list(children)    #save node children to generated child

        return children

    #calculates the score the distance from the starting point to the ending point (closer = better = larger number)
    def calcEvalScore(self,state):
        return self.maxDist - getHeuristic(state)

    #compares the score of 2 mcts nodes
    def __lt__(self, other):
        return self.q < other.q

    #print the score, node depth, and actions leading to it
    #for use with debugging
    def __str__(self):
        return str(self.q) + ", " + str(self.n) + ' - ' + str(self.getActions())


# Monte Carlo Tree Search Algorithm code
class MCTSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        #setup
        intializeDeadlocks(state)
        iterations = 0
        bestNode = None
        initNode = MCTSNode(state.clone(), None, None, getHeuristic(state))

        while(iterations < maxIterations):
            #print("\n\n---------------- ITERATION " + str(iterations+1) + " ----------------------\n\n")
            iterations += 1

            #mcts algorithm
            rollNode = self.treePolicy(initNode)
            score = self.rollout(rollNode)
            self.backpropogation(rollNode, score)

            #if in a win state, return the sequence
            if(rollNode.checkWin()):
                return rollNode.getActions()

            #set current best node
            bestNode = self.bestChildUCT(initNode)


            #if in a win state, return the sequence
            if(bestNode and bestNode.checkWin()):
                return bestNode.getActions()


        #return solution of highest scoring descendent for best node
        #if this line was reached, that means the iterations timed out before a solution was found
        return self.bestActions(bestNode)


    #returns the descendent with the best action sequence based
    def bestActions(self, node):
        #no node given - return nothing
        if node == None:
            return []

        bestActionSeq = []
        while(len(node.children) > 0):
            node = self.bestChildUCT(node)

        return node.getActions()


    ####  MCTS SPECIFIC FUNCTIONS BELOW  ####
    def treePolicy(self, rootNode):
        curNode = rootNode
        visited = []
        sc1=curNode.state.clone()
        if curNode.n ==0:
            return curNode
        while sc1.checkWin()==False:
            t = curNode.getChildren(visited)
            for i in t:
                if i.n==0:
                    return i
            curNode = self.bestChildUCT(curNode)
            if curNode!=None:
                sc1=curNode.state.clone()
            if curNode==None:
                return rootNode
        return curNode

    def bestChildUCT(self, node):
        c = 1
        bestChild = None
        bestAction=[-1,None]
        for i in node.children:
            t=i.state.clone()
            if t.checkWin()==True:
                return i
            while (i.n==0):
                return i
            r = (i.q/i.n) + (c*(math.sqrt((2*math.log(node.n))/i.n)))
            if r == bestAction[0]:
                bestAction[1]=i
            if r > bestAction[0]:
                bestAction = [r,i]
        if bestAction[1]==None:
            return node.parent
        else:
            return bestAction[1]

    def rollout(self,node):
        numRolls = 7
        t = node.state.clone()
        for i in range(numRolls):
            act = random.choice(directions)
            t.update(act['x'], act['y'])
            if (t.checkWin()):
                s = node.calcEvalScore(t)
                return s
        s = node.calcEvalScore(t)
        return s
        return 0

    def backpropogation(self, node, score):
        while(node!=None):
            node.n+=1
            node.q+=score
            node=node.parent
        return
