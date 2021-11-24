import numpy as np
import os  
import pdb
import sys
import random
import scipy
from itertools import permutations
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from Utility import *


def ClassifierEval(args, StatsPath, Score, X, p, qs):
    """
    Mutual information calculation
    StatsPath: str. Path to save
    X: array. Features;
    Score: class one posterial probability of classifier
    p: proportion of used queried labels
    qs: str. query strategy
    """
    Mean0 = np.zeros(args.FeatLen); Mean1 = np.zeros(args.FeatLen);
    Mean0[0] = Mean0[0] - args.Sep / 2; Mean1[0] = Mean1[0] + args.Sep / 2 
    Cov0 = np.diag(np.ones(args.FeatLen)); Cov1 = np.diag(np.ones(args.FeatLen) +args.Del)

    P0 = multivariate_normal.pdf(X, mean = Mean0, cov =Cov0);
    P1 = multivariate_normal.pdf(X, mean = Mean1, cov =Cov1);
   
    Pos0 = P0 * 0.5 / (P0 * 0.5 + P0 * 0.5)
    Pos1 = P1 * 0.5 / (P0 * 0.5 + P0 * 0.5)
    I = np.corrcoef(Score, Pos1)[0, 1]; 

    if os.path.isfile(StatsPath + '%s.xlsx'%qs):
        wb=load_workbook(StatsPath + '%s.xlsx'%qs);
        if not 'corr' in wb.sheetnames:
            ws1=wb.create_sheet("corr")
        else:
            ws1=wb["corr"]
    else:
        wb = Workbook(); ws1 = wb.create_sheet("corr");
    Alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    ws1.cell(row=args.Trial, column=p, value=I)
    ws1['%s%d'%(Alphabet[p-1], args.Trial + 1)] ='=AVERAGE(%s%d:%s%d)'%(Alphabet[p-1], 1, Alphabet[p-1], args.Trial)
    wb.save(StatsPath + '%s.xlsx'%qs)
    return I 

def GetBER(args, Mean):
    """
    numerically compute BER
    Mean: center of the second component
    """
    X = np.zeros(args.FeatLen); X[0] = 0; X[1:] = np.inf
    BER = multivariate_normal.cdf(X, mean = Mean)
    return BER

def GetData(Args):
    if Args.DataType == 'Syn':
        Mean0 = np.zeros(Args.FeatLen); Mean1 = np.zeros(Args.FeatLen);
        Mean0[0] = Mean0[0] - Args.Sep / 2; Mean1[0] = Mean1[0] + Args.Sep / 2 
        Cov0 = np.diag(np.ones(Args.FeatLen)); Cov1 = np.diag(np.ones(Args.FeatLen) +Args.Del)
        BER = GetBER(Args, Mean1) # numerically compute BER
        print('Creating synthetic dataset of two component gaussian. Sample Size: %d, Speration:%.2f, Delta:%2f, Dimension: %d, BER:%.2f'%(Args.S, Args.Sep, Args.Del, Args.FeatLen, BER))
        StatsPath = os.getcwd() + '/Stats/%s/Data/D%d/Sep%.2f/Delta%.2f/Size%d/'%(Args.DataType, Args.FeatLen, Args.Sep, Args.Del, Args.S)
       
        S1 = int(Args.S/2); 
        mn0 = multivariate_normal(Mean0, Cov0); 
        mn1 = multivariate_normal(Mean1,Cov1);
        trmn0 = multivariate_normal(Mean0, Cov0); 
        trmn1 = multivariate_normal(Mean1, Cov1);
        Feat0 = mn0.rvs(size=S1, random_state=Args.Trial); Feat1 = mn1.rvs(size=S1, random_state=Args.Trial + 1)
        TrFeat0 = trmn0.rvs(size=S1, random_state=Args.Trial+2); TrFeat1 = trmn1.rvs(size=S1, random_state=Args.Trial+3)

        Data0 = np.concatenate((Feat0, np.zeros((S1, 1))), 1); TrData0 = np.concatenate((TrFeat0, np.zeros((S1, 1))), 1)
        Data1 = np.concatenate((Feat1, np.ones((S1, 1))), 1); TrData1 = np.concatenate((TrFeat1, np.ones((S1, 1))), 1)
  
        Data = np.concatenate((Data0, Data1), 0); TrData = np.concatenate((TrData0, TrData1), 0);
        Data = np.random.RandomState(Args.Trial-1).permutation(Data); TrData = np.random.RandomState(Args.Trial-1).permutation(TrData)

        if not os.path.exists(StatsPath):
            os.makedirs(StatsPath)
        if Args.SaveData:
            # np.save(StatsPath + 'SynSep%.2fDel%.2fSize%d.npy'%(Args.Sep, Args.Del, Args.S), Data); 
            # np.save(StatsPath + 'TrSynSep%.2fDel%.2fSize%d.npy'%(Args.Sep, Args.Del, Args.S), TrData)
            DrawData(Args, TrData, BER)
     

    elif Args.DataType == 'MNIST':
        print('loading %s dataset'%Args.DataType)
        StatsPath = os.getcwd() + '/Stats/%s/Data/Feat1/'%(Args.DataType);
        # perm = list(permutations(np.arange(10),2))
        # SampleX = np.load(StatsPath + 'MNISTl%dR0.npy'%perm[Args.Trial - 1][0]); SampleY = np.load(StatsPath + 'MNISTl%dR0.npy'%perm[Args.Trial-1][1]); 
        # SampleX[:, -1] = 0; SampleY[:, -1] = 1; 
        TwoDigitId = random.sample(list(np.arange(10)), 2); TwoR = random.sample(list(np.arange(10)), 2)        
        # SampleX = np.load(StatsPath + 'MNISTl%dR%d.npy'%(perm[(Args.Trial - 1)%90][0], Args.Trial/90)); SampleY = np.load(StatsPath + 'MNISTl%dR%d.npy'%(perm[(Args.Trial - 1)%90][0], Args.Trial/90 + 1));
        # SampleY2 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(perm[(Args.Trial - 1)%90][1], Args.Trial/90 + 1)); QueryIndex = np.random.RandomState(Args.Trial-1).permutation(len(SampleY2))
        
        SampleX = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[0], TwoR[0])); SampleY = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[0], (TwoR[0] + 1) % 10));
        SampleY2 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[1], TwoR[0])); QueryIndex = np.random.RandomState(Args.Trial-1).permutation(len(SampleY2))
        SampleY[np.int64(QueryIndex[:300])] = SampleY2[np.int64(QueryIndex[:300])]; 
        SampleX[:, -1] = 0; SampleY[:, -1] = 1;

        TrSampleX = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[0], (TwoR[0] + 2) % 10)); TrSampleY = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[0], (TwoR[0] + 3) % 10));
        TrSampleY2 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[1], (TwoR[0] + 1) % 10)); QueryIndex = np.random.RandomState(Args.Trial-1).permutation(len(TrSampleY2))
        TrSampleY[np.int64(QueryIndex[:300])] = TrSampleY2[np.int64(QueryIndex[:300])]
        TrSampleX[:, -1] = 0; TrSampleY[:, -1] = 1; 

        Data = np.vstack((SampleX, SampleY));TrData = np.vstack((TrSampleX, TrSampleY));
        Data = np.random.RandomState(Args.Trial-1).permutation(Data); TrData = np.random.RandomState(Args.Trial-1).permutation(TrData)

    elif Args.DataType == 'MNISTNull':
        
        print('loading %s dataset'%Args.DataType)
        StatsPath = os.getcwd() + '/Stats/MNIST/Data/Feat1/';perm = list(permutations(np.arange(10),2))
        # SampleX = np.load(StatsPath + 'MNISTl%dR0.npy'%perm[Args.Trial - 1][0]); SampleY = np.load(StatsPath + 'MNISTl%dR0.npy'%perm[Args.Trial-1][1]); 
        # SampleX[:, -1] = 0; SampleY[:, -1] = 1; 
        # DigitId = (Args.Trial - 1)/10; R = (Args.Trial - 1) % 10
        # SampleX = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(perm[R][0]))); SampleY = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(perm[R][1]))); 
  
        DigitId = random.randint(0, 9); TwoR = random.sample(list(np.arange(10)), 2)
        SampleX = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(TwoR[0]))); SampleY = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(TwoR[1])));
        SampleX[:, -1] = 0; SampleY[:, -1] = 1;

        TrSampleX = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(TwoR[0] + 1)%10)); TrSampleY = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(TwoR[1] + 1)%10));
        TrSampleX[:, -1] = 0; TrSampleY[:, -1] = 1;
      
        Data = np.vstack((SampleX, SampleY));TrData = np.vstack((TrSampleX, TrSampleY));
        Data = np.random.RandomState(Args.Trial-1).permutation(Data); TrData = np.random.RandomState(Args.Trial-1).permutation(TrData)        
     
    elif Args.DataType == 'ADNI':
        print('loading %s dataset'%Args.DataType)
        StatsPath = os.getcwd() + '/Stats/%s/Data/'%(Args.DataType); Data = np.load(StatsPath + 'NomalizedData.npy')
        Data0 = Data[Data[:, -1] == 0]; Data1 = Data[Data[:, -1] == 1]; 
        QueryIndex0 = np.random.RandomState(Args.Trial-1).permutation(len(Data0))
        QueryIndex1 = np.random.RandomState(Args.Trial-1).permutation(len(Data1))
        

        TrSample0 = Data0[np.int64(QueryIndex0[:750])]; TrSample1 = Data1[np.int64(QueryIndex1[:250])];
        Sample0 = Data0[np.int64(QueryIndex0[750:])]; Sample1 = Data1[np.int64(QueryIndex1[250:500])];

        Data = np.vstack((Sample0, Sample1));TrData = np.vstack((TrSample0, TrSample1));
        Data = np.random.RandomState(Args.Trial-1).permutation(Data); TrData = np.random.RandomState(Args.Trial-1).permutation(TrData)
    return Data, TrData 