import argparse 
import numpy as np 

parser = argparse.ArgumentParser(description='Hypothesis testing with active learning')

parser.add_argument('--DataType', type = str, default = 'Syn', help = 'Data type')

parser.add_argument('--TestType', type = str, default = 'FR', help = 'Two-sammple test Type')

parser.add_argument('--qs', type = str, default = 'Bimodal', help = 'query scheme: Bimodal, Uncertainty, or Certainty' )

parser.add_argument('--cls', type = str, default = 'logistic', help = 'Classifier type: logistic, SVC, NN, CaliSVC or CaliNN')

parser.add_argument('--Sep', type = float, default = 1.5, help = 'Seperation between means')

parser.add_argument('--Del', type = float, default = 0, help = 'difference between variance between means')

parser.add_argument('--Per', type = float, default = 1, help = 'Proportion of unlabelled pool')

parser.add_argument('--BP', type = float, default = 0.2, help = 'Budget proportion')

parser.add_argument('--Interval', type = float, default = 0.01, help = 'Proportion interval')

parser.add_argument('--alpha', type = float, default = 0.05, help = 'significance level')

parser.add_argument('--FeatLen', type = int, default = 2, help = 'feature length')

parser.add_argument('--InitSize', type = int, default = 15, help = 'Initial size to train a classifier')

parser.add_argument('--MaxQ', type = int, default = 500, help = 'Maximum query complexity') # It is only for synthetic data under the null

parser.add_argument('--ChenM', type = int, default = 1, help = 'nearest number in Chen stats')

parser.add_argument('--LoadData', type = int, default = 0, help = 'Load existing data or not')

parser.add_argument('--S', type = int, default = 500, help = 'Sample size')

parser.add_argument('--SaveData', type = int, default = 0, help = 'Save data or not')

parser.add_argument('--Trial', type = int, default = 1, help = 'Trial number')

parser.add_argument('--PermuTrial', type = int, default = 1, help = 'Permutation trials')

parser.add_argument('--Plot_Stats', type = int, default = 0, help = 'Plot stats distribution or not')

parser.add_argument('--Plot_Dimension', type = int, default = 0, help = 'Plot dimension results or not')

parser.add_argument('--Plot_CI', type = int, default = 0, help = 'Plot confindence interval for Type I error or not')

parser.add_argument('--Plot_FD', type = int, default = 0, help = 'Plot f-divergence or not')

parser.add_argument('--Plot_FD2', type = int, default = 0, help = 'compute all f-divergence or not')

parser.add_argument('--Plot_Trend', type = int, default = 0, help = 'Plot all results trend or not')

parser.add_argument('--LoadQuery', type = int, default = 0, help = 'Load query index or not')

parser.add_argument('--RunTest', type = int, default = 0, help = 'Run the label-efficient two-sample test')
