import numpy as np 
import os 
import pdb
import random
import pickle
import sys
import time
import matlab.engine
from Dataset import * 
sys.path.append(os.getcwd() + "/Statsticscollection/")
from ComputeStatistics import *


def LabelEfficientTwoSample(args, Data, HoldoutData, StatsPath, FigurePath, QueryLoadPath):
	if args.RunTest == 1:

		# Create directory
		print('========================= Running label efficient two-sample test ==============================')
		if args.qs != 'Passive':
			QueryIndexPath = QueryLoadPath + 'QueryIndex(ClsUncertaintyQuery)/%s/%s/RunDistri/InitTrainingSize%d/'%(args.qs, args.cls, args.InitSize)
			DistriPath = StatsPath + 'FoundCutEdges(ClsUncertaintyQuery)/%s/%s/RunDistri/InitTrainingSize%d/'%(args.qs, args.cls, args.InitSize)
			DistriFigurePath = FigurePath + 'StatsDistribution(ClsUncertaintyQuery)/%s/%s/InitTrainingSize%d/'%(args.qs, args.cls, args.InitSize)			
		else:
			QueryIndexPath = QueryLoadPath + 'QueryIndex(ClsUncertaintyQuery)/%s/RunDistri/'%(args.qs)
			DistriPath = StatsPath + 'FoundCutEdges(ClsUncertaintyQuery)/%s/RunDistri/'%(args.qs)
			DistriFigurePath = FigurePath + 'StatsDistribution(ClsUncertaintyQuery)/%s/'%(args.qs)

		Per = np.arange(args.Interval, args.Per + args.Interval, args.Interval)
		if not os.path.exists(QueryIndexPath):
			os.makedirs(QueryIndexPath)
		if not os.path.exists(DistriPath):
			os.makedirs(DistriPath)
		if not os.path.exists(DistriFigurePath):
			os.makedirs(DistriFigurePath)
		# Whether or not to load the saved query index  

		if args.LoadQuery == 1 and os.path.isfile(QueryIndexPath + 'QueryIndex%d.txt'%(args.Trial)):
			with open(QueryIndexPath + 'QueryIndex%d.txt'%(args.Trial), 'rb') as fp: QueryIndex = pickle.load(fp)
			print('Load label query successfully!')

		PemuCutEdgeNum = np.zeros((2, len(Per)));

		# Extract features and labels 
		X = Data[:, : -1]; Y = Data[:, -1]; 
		
		# Query label
		if not(args.LoadQuery == 1 and os.path.isfile(QueryIndexPath + 'QueryIndex%d.txt'%(args.Trial))):
			# Bimodal query
			if  args.qs == 'Bimodal':
				QueryIndex = GetActiveClassifier2(args, Data, HoldoutData)	
			# Passive query
			elif args.qs == 'Passive':
				QueryIndex = np.random.permutation(len(Data));
			# Certainty query or uncertain query 
			elif (args.qs =='Certainty' or args.qs == 'Uncertainty'):
				QueryIndex = GetActiveClassifier(args, Data, HoldoutData)

		if not(args.LoadQuery == 1 and os.path.isfile(QueryIndexPath + 'QueryIndex%d.txt'%(args.Trial))):
			with open(QueryIndexPath + 'QueryIndex%d.txt'%(args.Trial), 'wb') as rp:
				pickle.dump(QueryIndex, rp)	

		print('maximum query complexity: %d'%len(QueryIndex))# temparory code
		if args.Plot_Stats==1:
			Per = np.arange(args.Interval, args.Per + args.Interval, args.Interval);QueryIndex=np.int64(QueryIndex)

			# Obtain statistis at each proportion of query 
			for u, p in enumerate(Per):
				# stop early for permutation test
				if p>args.BP and (args.TestType=='PermuFR' or args.TestType=='PermuChen' or args.TestType=='PermuHotelling'):
					break
				SubX = X[np.sort(QueryIndex[:int(len(QueryIndex) * p)])]; SubY = Y[np.sort(QueryIndex[:int(len(QueryIndex) * p)])]; 
				SubData = np.hstack((SubX, SubY.reshape((-1, 1))));	
				start = time.time()

				if args.TestType=='Hotelling':
					PemuCutEdgeNum[0, u], PemuCutEdgeNum[1, u], _ = Hotelling(args, SubX[SubY==0], SubX[SubY==1]);end = time.time()
				elif args.TestType=='Chen':
					PemuCutEdgeNum[0, u], PemuCutEdgeNum[1, u], _ = Chen(args, SubX[SubY==0], SubX[SubY==1], M = args.ChenM);end = time.time()
				elif args.TestType=='FR':
					PemuCutEdgeNum[0, u], PemuCutEdgeNum[1, u], _ = FR(args, SubX[SubY==0], SubX[SubY==1]);end = time.time()
				elif args.TestType == 'PermuChen':
					PemuCutEdgeNum[0, u], PemuCutEdgeNum[1, u] = PermuChen(args, SubX, SubY);end = time.time()
				elif args.TestType == 'PermuHotelling':
					PemuCutEdgeNum[0, u], PemuCutEdgeNum[1, u] = PermuHotelling(args, SubX, SubY);end = time.time()
				elif args.TestType=='PermuFR':
					PemuCutEdgeNum[0, u], PemuCutEdgeNum[1, u] = PermuFR(args, SubX, SubY);end = time.time()

				print('========== Test: %s, %.2f proportion, Reject:%d, Pvalue:%.8f, permutation trial %d, %.2fs elapsed =========='%(args.TestType, p, PemuCutEdgeNum[1, u], PemuCutEdgeNum[0, u], args.Trial, end -start))

			np.save(DistriPath + 'CutEdgeNumTrial%d.npy'%args.Trial, PemuCutEdgeNum)
			print('========================= ploting statistic distribution ==============================')
			PlotStatsDistri2(args, DistriPath, DistriFigurePath, 'ClsUncertaintyQuery')

	
