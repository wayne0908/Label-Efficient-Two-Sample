import numpy as np 
import sys
import pdb
import os 
import random 
from Options import * 
sys.path.append(os.getcwd() + "/CreateDataset/")
sys.path.append(os.getcwd() + "/Statsticscollection/")
from Dataset import * 
from CollectStatistics import *

def main():

	"""
	Set up parameters
	"""
	args = parser.parse_args()

	"""
	Set random seed 
	"""
	print('======================= Random number with seed %d ========================='%args.Trial)
	random.seed(args.Trial)
	np.random.seed(args.Trial)
	"""
	Build directory
	"""
	print('========================= Build directory ==============================')
	StatsPath, FigurePath, QueryLoadPath, QueryLoadFigPath = GetDirectory(args)

	"""
	Acquire data
	"""
	print('======================= Acquiring data ==============================')
	HoldoutData, TrData = GetData(args)


	"""
	Label efficient two-sample test
	"""
	LabelEfficientTwoSample(args, TrData, HoldoutData, StatsPath, FigurePath, QueryLoadPath)


	"""
	Plot F-divergence
	"""
	if args.Plot_FD == 1:
		PlotFDivergence(args, QueryLoadPath, QueryLoadFigPath)


	"""
	Compute F-divergence
	"""
	if args.Plot_FD2 == 1:
		PlotFDivergence2(args, TrData, QueryLoadPath)

	"""
	Plot Type I/II error
	"""
	if args.Plot_Trend == 1:
		PlotResultsTrend(args, StatsPath, FigurePath)

	"""
	PlotDimensionResults
	"""
	if args.Plot_Dimension == 1:
		PlotDimensionTrend(args, FigurePath)

	"""
	plot Type I error confidence itnerval 
	"""
	if args.Plot_CI == 1:
		PlotCI(args, StatsPath, FigurePath)

if __name__ == '__main__':
	main()