
################################################
#                                              #
#     LEVEL GENERATOR FOR SOKOBAN FRAMEWORK    #
#     Written by [Lakshmi Sanjana Kalidindi]   #
#         lk2849                               #
################################################


# import libraries
import random
from agent import DoNothingAgent, RandomAgent, BFSAgent, DFSAgent, AStarAgent, HillClimberAgent, GeneticAgent, MCTSAgent
from helper import *
import os

import numpy
import sokoban_template
import sys
import queue as que



# COMMAND LINE ARGS
EVAL_AGENT = "AStar"			#sokoban master agent to use as evaluator
SOLVE_ITERATIONS = 1250			#number of iterations to run agent for

MIN_SOL_LEN = 5				    #minimum length of solution
MAX_SOL_LEN = 50				#maximum length of solution

NUM_LEVELS = 20					#number of levels to generate
OUT_DIR = "assets/gen_levels"	#directory to output generated levels
LEVEL_PREFIX = "Level"			#prefix for the filename of the generated levels


NUM_BOXES = 2					#Number of boxes
MIN_G = 6
MAX_G = 10



# SOKOBAN ASCII CHARS KEY #
_player = "@"  #1 per game
_crate = "$"
_wall = "#"
_floor = " "
_emptyGoal = "."
_filledGoal = "*"


########################################################[ TEMPLATEs ]######################################################################

###############################################
#     A series of 17 templates                #
#                                             #
#  templates used to create sudo-random maps  #
###############################################


templates_pool = []

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 1
current_temp_grid[1][2] = 1
current_temp_grid[1][3] = 1
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 0
current_temp_grid[2][1] = 1
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 1
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 1
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 1
current_temp_grid[1][3] = 1
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 0
current_temp_grid[2][1] = 1
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 1
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 1
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 1
current_temp_grid[0][4] = 1

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 2
current_temp_grid[1][3] = 1
current_temp_grid[1][4] = 1

current_temp_grid[2][0] = 0
current_temp_grid[2][1] = 1
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 1
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 1
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 2
current_temp_grid[1][3] = 2
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 0
current_temp_grid[2][1] = 1
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 1
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 1
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 2
current_temp_grid[1][3] = 2
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 0
current_temp_grid[2][1] = 2
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 2
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 1
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 1
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 1
current_temp_grid[1][3] = 1
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 1
current_temp_grid[2][1] = 1
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 1
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 2
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 1
current_temp_grid[1][3] = 1
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 1
current_temp_grid[2][1] = 1
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 2
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 1
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 1
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 1
current_temp_grid[1][3] = 1
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 1
current_temp_grid[2][1] = 1
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 2
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 2
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 1
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 1
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 1
current_temp_grid[1][3] = 2
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 1
current_temp_grid[2][1] = 1
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 1

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 2
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 2
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 1
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 1
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 1
current_temp_grid[1][3] = 2
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 0
current_temp_grid[2][1] = 2
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 1

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 2
current_temp_grid[3][2] = 2
current_temp_grid[3][3] = 2
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 2
current_temp_grid[1][3] = 2
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 1
current_temp_grid[2][1] = 1
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 1

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 2
current_temp_grid[3][2] = 2
current_temp_grid[3][3] = 2
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 1
current_temp_grid[1][2] = 1
current_temp_grid[1][3] = 1
current_temp_grid[1][4] = 1

current_temp_grid[2][0] = 0
current_temp_grid[2][1] = 1
current_temp_grid[2][2] = 2
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 1

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 1
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 1
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 2
current_temp_grid[1][3] = 2
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 0
current_temp_grid[2][1] = 2
current_temp_grid[2][2] = 2
current_temp_grid[2][3] = 2
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 2
current_temp_grid[3][2] = 2
current_temp_grid[3][3] = 2
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 0
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 2
current_temp_grid[1][3] = 2
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 0
current_temp_grid[2][1] = 2
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 1
current_temp_grid[3][1] = 1
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 1
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 1
current_temp_grid[4][1] = 1
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 1
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 1
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 1
current_temp_grid[1][2] = 1
current_temp_grid[1][3] = 1
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 0
current_temp_grid[2][1] = 2
current_temp_grid[2][2] = 1
current_temp_grid[2][3] = 2
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 1
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 1
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 1
current_temp_grid[4][2] = 0
current_temp_grid[4][3] = 1
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 2
current_temp_grid[1][3] = 2
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 0
current_temp_grid[2][1] = 2
current_temp_grid[2][2] = 2
current_temp_grid[2][3] = 2
current_temp_grid[2][4] = 0

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 1
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 1
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 1
current_temp_grid[4][2] = 1
current_temp_grid[4][3] = 1
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

current_temp_grid = numpy.ndarray((5,5))

current_temp_grid[0][0] = 0
current_temp_grid[0][1] = 0
current_temp_grid[0][2] = 0
current_temp_grid[0][3] = 0
current_temp_grid[0][4] = 0

current_temp_grid[1][0] = 0
current_temp_grid[1][1] = 2
current_temp_grid[1][2] = 2
current_temp_grid[1][3] = 2
current_temp_grid[1][4] = 0

current_temp_grid[2][0] = 1
current_temp_grid[2][1] = 1
current_temp_grid[2][2] = 2
current_temp_grid[2][3] = 1
current_temp_grid[2][4] = 1

current_temp_grid[3][0] = 0
current_temp_grid[3][1] = 1
current_temp_grid[3][2] = 1
current_temp_grid[3][3] = 1
current_temp_grid[3][4] = 0

current_temp_grid[4][0] = 0
current_temp_grid[4][1] = 1
current_temp_grid[4][2] = 1
current_temp_grid[4][3] = 0
current_temp_grid[4][4] = 0

templates_pool.append(current_temp_grid)

#####################################################################################################################

#turns 2d array of a level into an ascii string
def lev2Str(l):
	s = ""
	for r in l:
		s += ("".join(r) + "\n")
	return s


# creates an empty base sokoban level
def makeEmptyLevel(w=9,h=9):
	l = []

	tbw = [] #top/bottom walls
	lrw = [] #left/right walls

	#initialize row setup
	for i in range(w):
		tbw.append(_wall)
	for i in range(w):
		if i == 0 or i == (w-1):
			lrw.append(_wall)
		else:
			lrw.append(_floor)

	#make level
	for i in range(h):
		if i == 0 or i == (h-1):
			l.append(tbw[:])
		else:
			l.append(lrw[:])

	return l


##########################################################[ HELPER FUNCTIONS ]###############################################################

def Level_val(level, part, gridX, gridY, h, w, size):
	cpy = True
	count = 0
	flag = 0
	for x in range(0,size):
		if (gridY+x) < w:
			if level[gridX][gridY+x] != 0:
				flag = 1
				if level[gridX][gridY+x] != part[0][x]:
					count = count + 1
					cpy = False
	
	for x in range(0,size):
		if (gridX+x) < h:
			if level[gridX+x][gridY] != 0:
				flag = 1
				if level[gridX+x][gridY] != part[x][0]:
					count = count + 1
					cpy = False

	for x in range(0,size):
		if (gridX+4) < h:
			if gridY+x < w:
				if level[gridX+4][gridY+x] != 0:
					flag = 1
					if level[gridX+4][gridY+x] != part[4][x]:
						count = count + 1
						cpy = False
	for x in range(0,size):
		if (gridY+4) < w:
			if gridX+x < h:
				if level[gridX+x][gridY+4] != 0:
					flag = 1
					if level[gridX+x][gridY+4] != part[x][4]:
						count = count + 1
						cpy = False

	return cpy

def create_emp_room(small, large):
	random.seed()
	gridX = 0
	w = random.randint(small, large)
	gridY = 0
	h = random.randint(small, large)
	level = numpy.ndarray((h,w))
	count = 0
	level.fill(0)
	while True:
		ix = random.randint(0,16)
		siz = 5
		part = templates_pool[ix]
		if Level_val(level, part, gridX, gridY, h, w, siz):
			for x in range(0,siz-2):
				for y in range(0,siz-2):
					if gridX+x < level.shape[0] and gridY+y < level.shape[1]:
						level[gridX+x][gridY+y] = part[x+1][y+1]
			gridX = gridX + 3

			if gridX > h-1:
				gridX = 0
				gridY = gridY + 3
		count = count + 1
		if gridY > w-1 or count > 100000:
			return level

def getGoodRoom(small, large):
	fnd = False
	while not fnd:
		level = create_emp_room(small,large)
		fnd = True
		flag1 = 1
		dp = numpy.ndarray((level.shape[0],level.shape[1]))
		count = 1
		queue = que.Queue()
		dp.fill(0)
		z0 = level.shape[0]
		z1 = level.shape[1]
		for x in range(0,z0):
			for y in range(0, z1):
				if level[x][y].astype(int) == 1:
					if dp[x][y] == 0:
						dp[x][y] = count
						flag1 = 0
						queue.put((x,y))
						while not queue.empty():
							i,j = queue.get()
							if i+1 < z0:
								if (level[i+1][j].astype(int) == 1):
									flag1 = 0
									if (dp[i+1][j] == 0):
										dp[i+1][j] = count
										queue.put((i+1,j))
							if (j+1 < z1):
								if (level[i][j+1].astype(int) == 1):
									flag1 = 0
									if (dp[i][j+1] == 0):
										dp[i][j+1] = count
										queue.put((i,j+1))
							if j-1 >= 0:
								if (level[i][j-1].astype(int) == 1):
									flag1 = 0
									if (dp[i][j-1] == 0):
										dp[i][j-1] = count
										queue.put((i,j-1))
							if i-1 >= 0:
								if (level[i-1][j].astype(int) == 1):
									flag1 = 0
									if (dp[i-1][j] == 0):
										dp[i-1][j] = count
										queue.put((i-1,j))

						count = count + 1

		for row in dp:
			for i in row:
				p = 0
				if i > 1:
					fnd = False

		if fnd == 1:
			if level.shape[0] < 3 or level.shape[1] < 3:
				p = p + 1
				fnd = True
			for x in range(0, level.shape[0]-2):
				for y in range(0, level.shape[1]-2):
					terminate_level = True
					if level[x][y].astype(int) == 1:
						p = p + 1
						for i in range(0,2):
							for j in range(0,2):
								if(not(level[x+i][y+j].astype(int) == 1)):
									p = p + 1
									terminate_level = False

						if terminate_level == 1:
							p = p + 1
							if x+3 < level.shape[0] and y+2 < level.shape[1]:
								if level[x+3][y] == 1 and level[x+3][y+1] == 1 and level[x+3][y+2] == 1:
									fnd =  False
							if y+3 < level.shape[1] and x+2 < level.shape[0]:
								if level[x][y+3] == 1 and level[x+1][y+3] == 1 and level[x+2][y+3] == 1:
									fnd =  False

			fnd = True
		if fnd == 1:
			for x in range(0,z0):
				for y in range(0, z1):
					if (level[x][y].astype(int) == 1):
						wc = 0
						if ((x+1) < z0):
							if (level[x+1][y].astype(int) == 2):
								p = 0
								wc = wc + 1
						else:
							wc = wc + 1

						if ((x-1) >= 0):
							if (level[x-1][y].astype(int) == 2):
								p = 0
								wc = wc + 1
						else:
							wc = wc + 1

						if ((y+1) < z1):
							if (level[x][y+1].astype(int) == 2):
								p = 0
								wc = wc + 1
						else:
							wc = wc + 1

						if ((y-1) >= 0):
							if (level[x][y-1].astype(int) == 2):
								p = 0
								wc = wc + 1
						else:
							wc = wc + 1

						if (wc > 2):
							p = 0
							fnd = False

	return level

def player(level):
	k = True
	p = 0
	while k == 1:
		x = random.randint(0,level.shape[0]-1)
		p = p + 1
		y = random.randint(0,level.shape[1]-1)
		if level[x][y] == 1:
			p = 0
			k = False
			level[x][y] = 3
	return level

def addGoal(level, numberOfGoals):
	p = 0
	k = True
	count = 0

	while k:
		p = p + 1
		x = random.randint(0,level.shape[0]-1)
		y = random.randint(0,level.shape[1]-1)

		if level[x][y] == 1:
			p = p + 1
			level[x][y] = 4
			numberOfGoals = numberOfGoals - 1

		if numberOfGoals <= 0:
			p = p + 1
			k = False

		count = count + 1
		if count > 10000:
			p = p + 1
			break

	return level

def add_a_box(level, noofbox):
	p = 0
	k = True
	count = 0

	while k:
		
		x = random.randint(0,level.shape[0]-2)
		p = p + 1
		y = random.randint(0,level.shape[1]-2)

		if level[x][y] == 1:
			level[x][y] = 5
			p = p + 1
			noofbox = noofbox - 1

		if noofbox <= 0:
			p = p + 1
			k = False

		count = count + 1
		p = p + 1
		if count > 10000:
			break

	return level

def write_level(level):
	p = 0
	for x in range(0,level.shape[1]+2):
		p = p + 1
		sys.stdout.write("#")
	print ("")

	for row in level:
		sys.stdout.write("#")
		for i in row:
			if i == 2:
				p = p + 1
				sys.stdout.write("#")
			elif i == 1:
				p = p + 1
				sys.stdout.write(" ")
			elif i == 3:
				p = p + 1
				sys.stdout.write("@")
			elif i == 4:
				p = p + 1
				sys.stdout.write(".")
			elif i == 5:
				p = p + 1
				sys.stdout.write("$")
		print("#")
	z1 = level.shape[1]
	for x in range(0,z1+2):
		sys.stdout.write("#")

def createString(level):
	string = ""
	flag = 0
	z1 = level.shape[1]
	for x in range(0,z1+2):
		string = string + "#"
	string = string + "\n"
	for row in level:
		string = string + "#"
		for i in row:
			flag = flag + 1 
			if i == 2:
				flag = flag + 1 
				string = string + "#"
			elif i == 1:
				flag = flag + 1 
				string = string + " "
			elif i == 3:
				flag = flag + 1 
				string = string + "@"
			elif i == 4:
				flag = flag + 1 
				string = string + "."
			elif i == 5:
				flag = flag + 1 
				string = string + "$"
		string = string + "#\n"
	for x in range(0,level.shape[1]+2):
		flag = flag + 1 
		string = string + "#"
	string = string + "\n"
	#flag val for tracing
	return string

def countFloorSpaces(level):
	floors = 0
	for ch in level:
		if (ch == ' '):
			p = 0
			floors = floors + 1
	return floors

def buildALevel():
	count = 0
	strLevel = ""
	obj = NUM_BOXES
	visited = set([strLevel])
	flag = 0
	level = getGoodRoom(MIN_G, MAX_G)
	maxObj = countFloorSpaces(createString(level))
	maxObj = maxObj / 2
	solver = None
	if EVAL_AGENT == 'DoNothing':
		solver = DoNothingAgent()
	elif EVAL_AGENT == 'Random':
		solver = RandomAgent()
	elif EVAL_AGENT == 'BFS':
		solver = BFSAgent()
	elif EVAL_AGENT == 'DFS':
		solver = DFSAgent()
	elif EVAL_AGENT == 'AStar':
		solver = AStarAgent()
	elif EVAL_AGENT == 'HillClimber':
		solver = HillClimberAgent()
	elif EVAL_AGENT == 'Genetic':
		solver = GeneticAgent()
	elif EVAL_AGENT == 'MCTS':
		solver = MCTSAgent()

	red_count = 1
	while (red_count!=0):
		#function calls
		tempLevel = level.copy()
		tempLevel = player(tempLevel)
		tempLevel = addGoal(tempLevel, obj)
		tempLevel = add_a_box(tempLevel, obj)

		strLevel =  createString(tempLevel)

		if strLevel not in visited:
			flag = 0
			visited.add(strLevel)	
			solvable, solLen = solveLevel(strLevel,solver)
			print(count," : ", solvable, "SolveLen :", solLen)
			if solvable:
				red_count = red_count - 1
				print(strLevel)
				return strLevel
			count = count + 1
			if count > 1000:
				count = 0
				obj = obj + 1
			if obj > maxObj:
				break
	return "Error!!!"

############################################################################################################################################

#use the agent to attempt to solve the level
def solveLevel(l,bot):
	#create new state from level
	state = State()
	state.stringInitialize(l.split("\n"))

	#evaluate level
	sol = bot.getSolution(state, maxIterations=SOLVE_ITERATIONS)
	for s in sol:
		state.update(s['x'],s['y'])
	return state.checkWin(), len(sol)


#generate and export levels using the PCG level builder and agent evaluator
def generateLevels():
	#set the agent
	solver = None
	if EVAL_AGENT == 'DoNothing':
		solver = DoNothingAgent()
	elif EVAL_AGENT == 'Random':
		solver = RandomAgent()
	elif EVAL_AGENT == 'BFS':
		solver = BFSAgent()
	elif EVAL_AGENT == 'DFS':
		solver = DFSAgent()
	elif EVAL_AGENT == 'AStar':
		solver = AStarAgent()
	elif EVAL_AGENT == 'HillClimber':
		solver = HillClimberAgent()
	elif EVAL_AGENT == 'Genetic':
		solver = GeneticAgent()
	elif EVAL_AGENT == 'MCTS':
		solver = MCTSAgent()

	#create the directory if it doesn't exist
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)

	#create levels
	totLevels = 0
	while totLevels < NUM_LEVELS:
		lvl = buildALevel()

		solvable, solLen = solveLevel(lvl,solver)

		#uncomment these lines if you want to see all the generated levels (including the failed ones)
		'''
		print(f"{lvl}solvable: {solvable}")
		if solvable:
			print(f"  -> solution len: {solLen}\n")
		else:
			print("")
		'''

		#export the level if solvable
		if solvable and solLen >= MIN_SOL_LEN and solLen <= MAX_SOL_LEN:
			with open(f"{OUT_DIR}/{LEVEL_PREFIX}_{totLevels}.txt",'w') as f:
				f.write(lvl)
			totLevels+=1

			#show the level exported
			print(f"LEVEL #{totLevels}/{NUM_LEVELS} -> {solLen} MOVES\n{lvl}")



#run whole script to generate
if __name__ == "__main__":
	generateLevels()
