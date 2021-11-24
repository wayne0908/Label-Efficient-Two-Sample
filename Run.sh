# Passive, Bimodal, Certainty, Uncertainty, 
for t in {1..200}
do 	
	python main.py --DataType='Syn' --Trial=$t --PermuTrial=500  --FeatLen=2 --Sep=1 --Del=0.6 --S=2000 --InitSize=50 --qs='Passive' --TestType='Chen' --alpha=0.05 --ChenM=1 --Interval=0.1 --Per=1 --cls='logistic' --RunTest=1 --LoadQuery=0 --Plot_Stats=1 --Plot_FD2=0 --Plot_Trend=0 	
	python main.py --DataType='Syn' --Trial=$t --PermuTrial=500  --FeatLen=2 --Sep=1 --Del=0.6 --S=2000 --InitSize=50 --qs='Bimodal' --TestType='Chen' --alpha=0.05 --ChenM=1 --Interval=0.1 --Per=1 --cls='logistic' --RunTest=1 --LoadQuery=0 --Plot_Stats=1 --Plot_FD2=0 --Plot_Trend=0
	python main.py --DataType='Syn' --Trial=$t --PermuTrial=500  --FeatLen=2 --Sep=1 --Del=0.6 --S=2000 --InitSize=50 --qs='Certainty' --TestType='Chen' --alpha=0.05 --ChenM=1 --Interval=0.1 --Per=1 --cls='logistic' --RunTest=1 --LoadQuery=0 --Plot_Stats=1 --Plot_FD2=0 --Plot_Trend=0
	python main.py --DataType='Syn' --Trial=$t --PermuTrial=500  --FeatLen=2 --Sep=1 --Del=0.6 --S=2000 --InitSize=50 --qs='Uncertainty' --TestType='Chen' --alpha=0.05 --ChenM=1 --Interval=0.1 --Per=1 --cls='logistic' --RunTest=1 --LoadQuery=0 --Plot_Stats=1 --Plot_FD2=0 --Plot_Trend=0
done