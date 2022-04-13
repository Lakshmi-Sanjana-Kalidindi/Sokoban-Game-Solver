import os
from sokoban import State
from helper import readLevel
from agent import DoNothingAgent, RandomAgent, BFSAgent, DFSAgent, AStarAgent, HillClimberAgent, GeneticAgent, MCTSAgent
import argparse
import random
from termcolor import colored

#have bot solve the game
def ai_play(lvlNumber, bot, maxIter):

	state = readLevel(lvlNumber)
	sol = bot.getSolution(state, maxIterations=maxIter)

	finish = False

	if len(sol) == 0:
		return colored("[ - ]", 'white'), 0

	#rollout the steps in the solution found
	try:
		for s in sol:
			state.update(s['x'],s['y'])

		#get status upon finish of the game
		if state.checkWin():
			return colored("[ O ]",'green'), 1
		else:
			return colored("[ ? ]",'magenta'), 0
	except:
		return colored("[ X ]",'red'), 0



if __name__ == '__main__':
	#establish parameters
	parser = argparse.ArgumentParser(description='PyGame wrapper for Sokoban')
	parser.add_argument('-a', '--agent', action='store', dest='agent', default='Random', help='Which agent algorithm to use (DoNothing, Random, BFS, DFS, AStar, HillClimber, Genetic, MCTS)')
	parser.add_argument('-i', '--iterations', action='store', dest='maxIter', default=3000, type=int, help='Number of iterations for the agent to search a solution for')

	args = parser.parse_args()

	print(" -- Key -- ")
	print(colored(" X  = code error","red"))
	print(colored(" -  = no sequence returned","white"))
	print(colored(" ?  = level unsolved","magenta"))
	print(colored(" O  = level solved","green"))
	print("")

	
	#set the agent
	solver = None
	if args.agent == 'DoNothing':
		solver = DoNothingAgent()
	elif args.agent == 'Random':
		solver = RandomAgent()
	elif args.agent == 'BFS':
		solver = BFSAgent()
	elif args.agent == 'DFS':
		solver = DFSAgent()
	elif args.agent == 'AStar':
		solver = AStarAgent()
	elif args.agent == 'HillClimber':
		solver = HillClimberAgent()
	elif args.agent == 'Genetic':
		solver = GeneticAgent()
	elif args.agent == 'MCTS':
		solver = MCTSAgent()
	else:
		print("!!! UNKNOWN AGENT '" + args.agent +"' !!!\n")
		parser.print_help()
		exit()

	print(f" -- SOLVING 100 LEVELS WITH [ {args.agent} ] AGENT FOR [ {args.maxIter} ] ITERATIONS -- ")


	#evaluate all of the levels
	solveAcc = 0
	for levelNo in range(100):
		result, win = ai_play(levelNo, solver, args.maxIter)
		print(f"Level <{levelNo}>\t{result}")
		if win == 1:
			solveAcc += 1

	print("")
	sacc = solveAcc / 100.0
	print(f"Levels solved: {sacc}%") 
	print(" -- ")




