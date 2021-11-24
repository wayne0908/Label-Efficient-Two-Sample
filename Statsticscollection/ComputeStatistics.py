import numpy as np 
import multiprocessing as mp
import pdb
import os
import pickle
import random
import typer 
import torch
from torch import nn
from skorch import NeuralNetClassifier
from scipy.stats import multivariate_normal
from Utility import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy.sparse import csr_matrix 
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn import svm
from sklearn.svm import SVC
from scipy.stats import chi2, norm
from scipy.sparse.csgraph import minimum_spanning_tree
from modAL.models import ActiveLearner
from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.stats import f
from sklearn.calibration import CalibratedClassifierCV
import matlab.engine


def GetPosterior(args, X, Y, cluster_num=16, option = 'Emp'):
	"""
	Get f-divergence. If data is synthetic, we use close-form posterior probability otherwise
	we use empirical probabilty.
	"""
	# closed form estimation
	if option == 'Closed':
		Mean0 = np.zeros(args.FeatLen); Mean1 = np.zeros(args.FeatLen);
		Mean0[0] = Mean0[0] - args.Sep / 2; Mean1[0] = Mean1[0] + args.Sep / 2 
		Cov0 = np.diag(np.ones(args.FeatLen)); Cov1 = np.diag(np.ones(args.FeatLen) +args.Del)

		P0 = multivariate_normal.pdf(X, mean = Mean0, cov =Cov0);
		P1 = multivariate_normal.pdf(X, mean = Mean1, cov =Cov1);    
		
		Pos0 = P0 * 0.5 / (P0 * 0.5 + P1 * 0.5)
		Pos1 = P1 * 0.5 / (P0 * 0.5 + P1 * 0.5)
	# empirical estimation
	elif option == 'Emp':
	    kmeans = KMeans(n_clusters=cluster_num, random_state=args.Trial).fit(X); Label = kmeans.labels_
	    Pos0 = np.zeros(len(X)); Pos1 = np.zeros(len(X));
	    for i in range(cluster_num):
	        for j in range(2):
	            if j == 0:
	            	Pos0[Label==i] = np.sum(Y[Label==i] == j)/sum(Label==i)
	            else:
	            	Pos1[Label==i] = np.sum(Y[Label==i] == j)/sum(Label==i)

	R = 2 * np.mean(Pos0) * np.mean(Pos1)
	
	return Pos0, Pos1, R 

def MIStats(args, X, Y, cluster_num=16):
    """
    Mutual information calculation
    StatsPath: str. Path to save
    X: array. Features; Y: array. Labels
    p: proportion of used queried labels
    qs: str. query strategy
    """
    if args.DataType == 'Syn':
    	Pos0, Pos1, R = GetPosterior(args, X, Y); u = np.mean(Pos0); v = np.mean(Pos1)
    	I = -(u * np.log(u) + v * np.log(v)) - (-np.mean(Pos0 * np.log(Pos0) + Pos1 * np.log(Pos1)))
    else:
	    kmeans = KMeans(n_clusters=cluster_num, random_state=args.Trial).fit(X); Label = kmeans.labels_
	    N = len(X); I = 0
	    for i in range(cluster_num):
	        for j in range(2):
	            p_xy = np.sum(Y[Label==i] == j)/N; p_x = np.sum(Label==i)/N; p_y = np.sum(Y==j)/N 
	            # print(p_xy, p_x, p_y)
	            if p_xy!=0:
	                I+=p_xy * np.log(p_xy/(p_x * p_y))
    return I
  

def FDStats2(args, X, Y):
	"""
	non-parametric method to compute F-divergence
	"""
	P, reject, S = FR(args, X, Y);
	I = -S/np.sqrt(len(X) + len(Y))
	
	return I, S 

def Hotelling(arg, X, Y):
	nx, p = X.shape
	ny, _ = Y.shape
	delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
	Sx = np.cov(X, rowvar=False)
	Sy = np.cov(Y, rowvar=False)
	S_pooled = ((nx-1)*Sx + (ny-1)*Sy)/(nx+ny-2)
	t_squared = (nx*ny)/(nx+ny) * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S_pooled)), delta)
	statistic = t_squared * (nx+ny-p-1)/(p*(nx+ny-2))
	F = f(p, nx+ny-p-1)
	pvalue = 1 - F.cdf(statistic)
	if pvalue <arg.alpha:
		reject = 1
	else:
		reject = 0
	return pvalue, reject, statistic

"""
Generate p-value from permutation distribution
PermuStats: permutation statistic. The last entry is the 
			statistic from the observed data. 
side: 'left' or 'right' for the observed statistic
"""
def PermuPvalue(args, PermuStats, side = 'left'):
	NullStats = PermuStats[:-1]; ObservedStats = PermuStats[-1] 
	Fig = plt.figure();ax = plt.gca()
	T = ax.hist(NullStats, 50, density=True, facecolor='blue'); 
	p=T[0]; bins=T[1]; p2 = np.hstack((0, p));
	if side == 'left':
		binnum = sum(bins <= ObservedStats)
	elif side == 'right': 
		binnum = sum(bins >= ObservedStats)
	
	if binnum == len(bins):
		pvalue= 1
	elif side == 'left':
		pvalue = np.sum(p2[:binnum+1] * (bins[-1] - bins[0])/50); 
	elif side == 'right':
	 	pvalue = np.sum(p2[-binnum:] * (bins[-1] - bins[0])/50); 

	reject = pvalue <=args.alpha; 
	plt.close('all')
	return pvalue, reject

def PermuHotelling(args, X, Y):
	PermuStats = np.zeros(args.PermuTrial + 1); total = 0
	with typer.progressbar(range(args.PermuTrial)) as progress:
		for value in progress:	
			PermutedY = np.random.permutation(Y); 
			_, _, PermuStats[total] = Hotelling(args, X[PermutedY==0], X[PermutedY==1])
			total+=1
	typer.echo(f"Processed {total} permutation.")
	_, _, PermuStats[-1] = Hotelling(args, X[Y==0], X[Y==1])
	pvalue, reject = PermuPvalue(args, PermuStats, side = 'right')
	return pvalue, reject

	
def FR(arg, X, Y, show = True):
	"""
	Fridman Rafsky statstic
	"""

	"""
	Construct minimum spanning tree
	"""
	NumCores = mp.cpu_count();
	Sample0 = np.hstack((X, np.zeros((len(X), 1)))); Sample1 = np.hstack((Y, np.ones((len(Y), 1))))
	Data = np.vstack((Sample0,Sample1)); GraphMatrix = pairwise_distances(Data[:, :-1], n_jobs = NumCores)
	for i in range(len(GraphMatrix)):
		GraphMatrix[i, :i + 1] = 0
	Tcsr = minimum_spanning_tree(csr_matrix(GraphMatrix))	
	Tree = Tcsr.toarray()
	for i in range(len(Tree)):
		Tree[np.where(Tree[i] != 0), i] = Tree[i][np.where(Tree[i] != 0)]
	BaseG = Tree.astype(bool).astype(int); 

	"""
	Cut-edge count
	"""
	R = 0; G = 0; n = len(X); m = len(Y); N = m + n; 
	for i in range(len(BaseG)):
		for j in range(i + 1, len(BaseG)):
			if BaseG[i, j] == 1:
				G+=1
				if Data[i, -1] != Data[j, -1]:
					R+=1
	"""
	Number of shared edges
	"""
	C=0
	for i in range(len(BaseG)):
		C+=0.5 * sum(BaseG[i] == 1) ** 2
	C-=G

	E = 2 * m * n/N + 1;
	Var = 2 * m * n / (N*(N-1)) *((2*m*n - N)/N + (C-N+2)/((N-2) * (N-3))*(N * (N-1) - 4*m*n + 2))
	S = (R - E) / np.sqrt(Var)

	"""
	P-value
	"""
	P = norm.cdf(S); 
	if P < arg.alpha:
		reject = 1
	else:
		reject = 0
	return P, reject, S

def PermuFR(args, X, Y):
	PermuStats = np.zeros(args.PermuTrial + 1); total = 0
	with typer.progressbar(range(args.PermuTrial)) as progress:
		for value in progress:	
			PermutedY = np.random.permutation(Y); 
			_, _, PermuStats[total] = FR(args, X[PermutedY==0], X[PermutedY==1])
			total+=1
	typer.echo(f"Processed {total} permutation.")
	_, _, PermuStats[-1] = FR(args, X[Y==0], X[Y==1])
	pvalue, reject = PermuPvalue(args, PermuStats, side = 'left')
	return pvalue, reject


def Chen(arg, X, Y, M = 1):
	"""
	Chen's statstic: from paper a new graph-based...
	"""
	"""
	Construct minimum spanning tree
	"""

	NumCores = mp.cpu_count();
	Sample0 = np.hstack((X, np.zeros((len(X), 1)))); Sample1 = np.hstack((Y, np.ones((len(Y), 1))));
	Data = np.vstack((Sample0,Sample1)); GraphMatrix = pairwise_distances(Data[:, :-1], n_jobs = NumCores)
	
	for i in range(len(GraphMatrix)):
		Indx = GraphMatrix[i].argsort()
		GraphMatrix[i, Indx[:M]] = np.inf 

	for i in range(len(GraphMatrix)):
		GraphMatrix[i, :i + 1] = 0
	Tcsr = minimum_spanning_tree(csr_matrix(GraphMatrix))	
	Tree = Tcsr.toarray()
	for i in range(len(Tree)):
		Tree[np.where(Tree[i] != 0), i] = Tree[i][np.where(Tree[i] != 0)]
	BaseG = Tree.astype(bool).astype(int); 

	"""
	Edge count
	"""
	R1 = 0; R2 = 0; G=0; n = len(X); m = len(Y); N = m + n
	for i in range(len(BaseG)):
		for j in range(i + 1, len(BaseG)):
			if BaseG[i, j] == 1:
				G+=1
				if Data[i, -1]==0 and Data[j, -1]==0: 
					R1+=1; 	
				elif Data[i, -1] == 1 and Data[j, -1]==1:
					R2+=1
	mu1 = G*n*(n-1)/(N*(N-1)); mu2 = G*m*(m-1)/(N*(N-1));

	"""
	Number of shared edges
	"""
	C=0
	for i in range(len(BaseG)):
		C+=0.5 * sum(BaseG[i] == 1) ** 2
	# C-=G
	C-=(N-1)
	"""
	Covariance count
	"""
	Delta11 = mu1 *(1-mu1) + 2 * C * n * (n-1) * (n-2) /(N * (N-1) * (N-2)) +\
	         (G * (G-1)-2 * C) * n * (n-1) * (n-2) * (n-3) / (N * (N - 1) * (N - 2) * (N-3));
	Delta22 = mu2 *(1-mu2) + 2 * C * m * (m-1) * (m-2) /(N * (N-1) * (N-2)) +\
	         (G * (G-1)-2 * C) * m * (m-1) * (m-2) * (m-3) / (N * (N - 1) * (N - 2) * (N-3));	
	Delta12 = (G * (G - 1) - 2 * C) * n * m * (n-1) * (m-1)/(N * (N - 1) * (N - 2) * (N -3)) - mu1 * mu2;
	Delta21 = Delta12 

	A = np.array((R1 - mu1, R2 - mu2)).reshape((1 , 2));
	B = np.linalg.inv(np.array(((Delta11, Delta12),(Delta21, Delta22))));
	A2 = A.reshape((2,1)); S = np.matmul(np.matmul(A, B), A2);
	
	"""
	P-value
	"""
	P = 1 - chi2.cdf(S, df = 2)[0,0]; 
	if P < arg.alpha:
		reject = 1
	else:
		reject = 0
	return P, reject, S

def PermuChen(args, X, Y):
	PermuStats = np.zeros(args.PermuTrial + 1); total = 0
	with typer.progressbar(range(args.PermuTrial)) as progress:
		for value in progress:	
			PermutedY = np.random.permutation(Y); 
			_, _, PermuStats[total] = Chen(args, X[PermutedY==0], X[PermutedY==1], M = args.ChenM)
			total+=1
	typer.echo(f"Processed {total} permutation.")
	_, _, PermuStats[-1] = Chen(args, X[Y==0], X[Y==1], M = args.ChenM)
	pvalue, reject = PermuPvalue(args, PermuStats, side = 'right')
	return pvalue, reject


def GetPassiveClassifier(args, Trdata, TestFeat):
	"""
	Classifier for passive learning
	"""
	X = Trdata[:, :-1]; Y = Trdata[:, -1]
	if args.cls=='knn':
		Cls = KNeighborsClassifier(n_neighbors=5); 
		Cls.fit(X, Y); Score = Cls.predict_proba(TestFeat)[:, -1]
	elif args.cls=='logistic':
		Cls = LogisticRegression(random_state=args.Trial).fit(X, Y)
		Score = Cls.predict_proba(TestFeat)[:, -1]
	elif args.cls=='NuSVC':
		Cls = svm.NuSVC(gamma='auto', random_state=args.Trial).fit(X, Y)
		Score = Cls.predict_proba(TestFeat)[:, -1]
	elif args.cls == 'SVC':
		Cls = SVC(gamma='auto', random_state=args.Trial, probability = True).fit(X, Y)
		Score = Cls.predict_proba(TestFeat)[:, -1]
	elif args.cls =='NN':
		Cls = BuildNN(len(X)); Cls.fit(X,Y)
		Score = Cls.predict_proba(TestFeat)[:, -1]
	return Cls, Score

class Torch_Model(nn.Module):
    def __init__(self, W):
        super(Torch_Model, self).__init__()
        self.W = W
        self.fcs = nn.Sequential(
                                nn.Linear(self.W,32),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(32,2),
        )

    def forward(self, x):
        out = x.float()
        out = out.view(-1,self.W)
        out = self.fcs(out)
        return out

def BuildNN(W=20):
	"""
	Neural network classifier
	W: int. Width of first layer
	"""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	Cls = NeuralNetClassifier(Torch_Model(W),
							  criterion=nn.CrossEntropyLoss,
							  optimizer=torch.optim.Adam,
							  train_split=None,
							  verbose=0,
							  device="cpu")
	return Cls
def Certainty(classifier, X_pool):
	
	Prob = classifier.predict_proba(X_pool)[:, -1]; uncertainty = np.abs(Prob - 0.5)
	query_idx = np.argmax(uncertainty)
	return query_idx, X_pool[query_idx]


def Bimodal(classifier, X_pool):
	
	Prob = classifier.predict_proba(X_pool)[:, -1]; 
	query_idx0 = np.argmax(Prob); query_idx1 = np.argmin(Prob)
	query_idx = [query_idx0, query_idx1]; 
	return query_idx, X_pool[query_idx]


def GetActiveClassifier(args, Trdata, HoldoutData):
	"""
	Classifier for active learning
	"""
	QueryIndex = np.zeros(len(Trdata)); PoolData = np.copy(Trdata);

	X_pool = PoolData[:, :-1]; y_pool = PoolData[:, -1]; Per = np.arange(args.Interval, args.Per + args.Interval, args.Interval);
	RandomIndex = np.random.permutation(len(Trdata))[:args.InitSize];InitTrX = X_pool[RandomIndex];InitTrY = y_pool[RandomIndex];
	QueryIndex[:args.InitSize] = RandomIndex; X_pool, y_pool = np.delete(X_pool, RandomIndex, axis=0), np.delete(y_pool, RandomIndex)
	HoldoutFeat = HoldoutData[:, :-1]
	Score = np.zeros(len(Per) - 1); Count = 0

	if args.cls=='logistic':
		Cls = LogisticRegression(random_state=args.Trial)
	elif args.cls=='SVC':
		Cls = SVC(gamma='auto', random_state=args.Trial, probability = True)
	elif args.cls == 'NN':
		Cls = BuildNN(len(X_pool[0])); 
		X_pool = torch.tensor(X_pool).float(); y_pool = torch.tensor(y_pool.reshape(-1)).long();
		HoldoutFeat = torch.tensor(HoldoutFeat).float();
		InitTrX = torch.tensor(InitTrX).float(); InitTrY = torch.tensor(InitTrY.reshape(-1)).long()
	elif args.cls == 'CaliNN':
		Base_Cls = BuildNN(len(X_pool[0])); 
		Cls = CalibratedClassifierCV(base_estimator=Base_Cls,n_jobs=mp.cpu_count(), ensemble=False)
		X_pool = torch.tensor(X_pool).float(); y_pool = torch.tensor(y_pool.reshape(-1)).long();
		HoldoutFeat = torch.tensor(HoldoutFeat).float();
		InitTrX = torch.tensor(InitTrX).float(); InitTrY = torch.tensor(InitTrY.reshape(-1)).long()
	elif args.cls=='CaliSVC':
		Base_Cls = SVC(random_state=args.Trial, gamma='auto', probability = True)
		Cls = CalibratedClassifierCV(base_estimator=Base_Cls, n_jobs=mp.cpu_count(), ensemble=False); 
	# Choose certainty query or uncertainty query
	if args.qs == 'Certainty':
		learner = ActiveLearner(estimator=Cls, query_strategy=Certainty, X_training=InitTrX, y_training=InitTrY)
	elif args.qs == 'Uncertainty':
		learner = ActiveLearner(estimator=Cls, X_training=InitTrX, y_training=InitTrY)
	
	Count2=args.InitSize;
	for i in range(args.InitSize, len(Trdata)):
		# active query
		query_index, query_instance = learner.query(X_pool); 
		Count2+=1; QueryIndex[i] = query_index
		X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
		X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

	return QueryIndex

def GetActiveClassifier2(args, Trdata, HoldoutData):
	"""
	Classifier for active learning
	"""
	QueryIndex = np.zeros(len(Trdata)); PoolData = np.copy(Trdata);
	X_pool = PoolData[:, :-1]; y_pool = PoolData[:, -1]; Per = np.arange(args.Interval, args.Per + args.Interval, args.Interval);
	HoldoutFeat = HoldoutData[:, :-1]
	RandomIndex = np.random.permutation(len(Trdata))[:args.InitSize];InitTrX = X_pool[RandomIndex];InitTrY = y_pool[RandomIndex];
	QueryIndex[:args.InitSize] = RandomIndex; X_pool, y_pool = np.delete(X_pool, RandomIndex, axis=0), np.delete(y_pool, RandomIndex)
	Count = 0

	if args.cls=='logistic':
		Cls = LogisticRegression(random_state=args.Trial)
	elif args.cls=='SVC':
		Cls = SVC(random_state=args.Trial, gamma='auto', probability = True)
	elif args.cls == 'NN':
		Cls = BuildNN(len(X_pool[0])); 
		X_pool = torch.tensor(X_pool).float(); y_pool = torch.tensor(y_pool.reshape(-1)).long();
		HoldoutFeat = torch.tensor(HoldoutFeat).float(); 
		InitTrX = torch.tensor(InitTrX).float(); InitTrY = torch.tensor(InitTrY.reshape(-1)).long() 
	elif args.cls == 'CaliNN':
		Base_Cls = BuildNN(len(X_pool[0])); 
		Cls = CalibratedClassifierCV(base_estimator=Base_Cls, n_jobs=mp.cpu_count(), ensemble=False); 
		X_pool = torch.tensor(X_pool).float(); y_pool = torch.tensor(y_pool.reshape(-1)).long();
		HoldoutFeat = torch.tensor(HoldoutFeat).float();
		InitTrX = torch.tensor(InitTrX).float(); InitTrY = torch.tensor(InitTrY.reshape(-1)).long()		
	elif args.cls=='CaliSVC':
		Base_Cls = SVC(random_state=args.Trial, gamma='auto', probability = True)
		Cls = CalibratedClassifierCV(base_estimator=Base_Cls, n_jobs=mp.cpu_count(), ensemble=False); 
	

	learner = ActiveLearner(estimator=Cls, query_strategy=Bimodal, X_training=InitTrX, y_training=InitTrY)

	Count2=args.InitSize;

	while(len(y_pool) > 0):
		query_index, query_instance = learner.query(X_pool);
		# print(Count2, len(X_pool))
		if len(query_index) ==1:
			QueryIndex[Count2] = query_index[0];Count2+=1
			X, y = X_pool[query_index[0]].reshape(1, -1), y_pool[query_index[0]].reshape(1, ); 
		else:
			if query_index[0] == query_index[1]:
				QueryIndex[Count2] = query_index[0];Count2+=1
				X, y = X_pool[query_index[0]].reshape(1, -1), y_pool[query_index[0]].reshape(1, );
			else:
				QueryIndex[Count2] = query_index[0]; QueryIndex[Count2+1] = query_index[1];
				Count2+=2
				X1, y1 = query_index[0].reshape(1, -1), y_pool[query_index[0]].reshape(1, ); 
				X2, y2 = query_index[1].reshape(1, -1), y_pool[query_index[1]].reshape(1, ); 

		X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

	return QueryIndex

