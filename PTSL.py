# -*- coding: utf-8 -*-

general_path = "Enter the path to the folder where this file is in."

import numpy as np
from random import gauss
from itertools import product
import pandas as pd
import matplotlib.pyplot as mpl

#———————————————————————————————————————
def main(expected = [5], sigma = 1, half_time = [5], rPT=np.linspace(0,10,21), rSLm=np.linspace(0,10,21), seed=0):
    count=0
    for prod_ in product(expected,half_time): #data = Forecast ; Halflife = [5] only one number otherwise change function such that output safe to each variable
        count+=1
        coeffs={'forecast':prod_[0],'hl':prod_[1],'sigma':sigma}
        print(prod_)
        output=batch(coeffs,rPT=rPT,rSLm=rSLm,nIter=100000,maxHP=100, seed = seed)
    return output


def batch(coeffs,nIter=1e5,maxHP=100,rPT=np.linspace(.5,10,20),
    rSLm=np.linspace(.5,10,20),seed=0):
    phi,output1=2**(-1./coeffs['hl']),[]
    for comb_ in product(rPT,rSLm):
        output2=[]
        for iter_ in range(int(nIter)):
            p,hp,count=seed,0,0
            while True:
                p=(1-phi)*coeffs['forecast']+phi*p+coeffs['sigma']*gauss(0,1) #forecast the price
                cP=p-seed;hp+=1 #change in price cP is new price - start price
                if cP>comb_[0] or cP<-comb_[1] or hp>maxHP:
                    output2.append(cP)
                    break
        mean,std=np.mean(output2),np.std(output2)
        print comb_[0],comb_[1],mean,std,mean/std
        output1.append((comb_[0],comb_[1],mean,std,mean/std))
    return output1

def plotCorrMatrix(path,corr,labels=None):
    # Heatmap of the correlation matrix
    if labels is None:labels=[]
    mpl.figure(num=None, figsize=(13, 10), dpi=600)
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.xticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.savefig(path)
    mpl.clf();mpl.close() # reset pylab
    return








  
data = pd.read_csv(general_path+"/SMI.csv", index_col=0)
SMI_changes= data[".SSMI"] - data[".SSMI"].shift(1)
SMI_sd_change = SMI_changes.std()
SMI_last_value = data[".SSMI"].tail(1).iloc[0]
#SMI plots
#Expected Value 0, Half-life 5
trading_rule_results = pd.DataFrame(main(expected = [9590.62], sigma = SMI_sd_change, half_time = [10],rPT=np.linspace(0,160,21), rSLm = np.linspace(0,160,21), seed = SMI_last_value))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/SMI_5.png",plotdata_pivoted, labels = plotdata_pivoted.columns)




#Plots from Book of Lopez
trading_rule_results = pd.DataFrame(main(expected = stock_data, sigma = stock_vola, half_time = [5]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/Testplot.png",plotdata_pivoted, labels = plotdata_pivoted.columns)


#Expected Value 0, Half-life 5
trading_rule_results = pd.DataFrame(main(expected = [0], sigma = 1, half_time = [5]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/0_5.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a') #=Beep sound

#Expected Value 0, Half-life 10
trading_rule_results = pd.DataFrame(main(expected = [0], sigma = 1, half_time = [10]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/0_10.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 0, Half-life 25
trading_rule_results = pd.DataFrame(main(expected = [0], sigma = 1, half_time = [25]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/0_25.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 0, Half-life 50
trading_rule_results = pd.DataFrame(main(expected = [0], sigma = 1, half_time = [50]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/0_50.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 0, Half-life 100
trading_rule_results = pd.DataFrame(main(expected = [0], sigma = 1, half_time = [100]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/0_100.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 5, Half-life 5
trading_rule_results = pd.DataFrame(main(expected = [5], sigma = 1, half_time = [5]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/5_5.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

#Expected Value 5, Half-life 10
trading_rule_results = pd.DataFrame(main(expected = [5], sigma = 1, half_time = [10]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/5_10.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 5, Half-life 25
trading_rule_results = pd.DataFrame(main(expected = [5], sigma = 1, half_time = [25]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/5_25.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 5, Half-life 50
trading_rule_results = pd.DataFrame(main(expected = [5], sigma = 1, half_time = [50]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/5_50.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 5, Half-life 100
trading_rule_results = pd.DataFrame(main(expected = [5], sigma = 1, half_time = [100]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/5_100.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 10, Half-life 10
trading_rule_results = pd.DataFrame(main(expected = [10], sigma = 1, half_time = [10]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/10_10.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 10, Half-life 10
trading_rule_results = pd.DataFrame(main(expected = [10], sigma = 1, half_time = [10]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/10_10.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 10, Half-life 25
trading_rule_results = pd.DataFrame(main(expected = [10], sigma = 1, half_time = [25]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/10_25.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 10, Half-life 50
trading_rule_results = pd.DataFrame(main(expected = [10], sigma = 1, half_time = [50]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/10_50.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value 10, Half-life 100
trading_rule_results = pd.DataFrame(main(expected = [10], sigma = 1, half_time = [100]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/10_100.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

#Expected Value -5, Half-life 5
trading_rule_results = pd.DataFrame(main(expected = [-5], sigma = 1, half_time = [5]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/m5_5.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value -5, Half-life 10
trading_rule_results = pd.DataFrame(main(expected = [-5], sigma = 1, half_time = [10]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/m5_10.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value -5, Half-life 25
trading_rule_results = pd.DataFrame(main(expected = [-5], sigma = 1, half_time = [25]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/m5_25.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value -5, Half-life 50
trading_rule_results = pd.DataFrame(main(expected = [-5], sigma = 1, half_time = [50]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/m5_50.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value -5, Half-life 100
trading_rule_results = pd.DataFrame(main(expected = [-5], sigma = 1, half_time = [100]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/m5_100.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value -10, Half-life 5
trading_rule_results = pd.DataFrame(main(expected = [-10], sigma = 1, half_time = [5]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/m10_5.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value -10, Half-life 10
trading_rule_results = pd.DataFrame(main(expected = [-10], sigma = 1, half_time = [10]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/m10_10.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value -10, Half-life 25
trading_rule_results = pd.DataFrame(main(expected = [-10], sigma = 1, half_time = [25]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/m10_25.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value -10, Half-life 50
trading_rule_results = pd.DataFrame(main(expected = [-10], sigma = 1, half_time = [50]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/m10_50.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')

#Expected Value -10, Half-life 100
trading_rule_results = pd.DataFrame(main(expected = [-10], sigma = 1, half_time = [100]))
plotdata = trading_rule_results[[0,1,4]]
plotdata_pivoted = plotdata.pivot(index = 1, columns = 0, values = 4)
plotCorrMatrix(general_path+"/m10_100.png",plotdata_pivoted, labels = plotdata_pivoted.columns)

print('\a')