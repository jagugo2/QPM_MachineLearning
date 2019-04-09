# coding: utf-8
# Your code here!
# All code based on Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Hoboken: Wiley. and
# Bailey, D. & Lopez de Prado, M. (2013). "An Open-Source Implementation of the Critical-Line Algorithm for Portfolio Optimization."
# Algorithms, Vol. 6, Issue 1, pp. 169-196.
# Slight modifications have been made to apply the code to SMI data.

general_path = "Enter the path to the folder where this file is in."
  
#Import Modules
import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch,random,numpy as np,pandas as pd
import imp
CLA = imp.load_source('CLA', general_path+ "CLA.py")

#———————————————————————————————————————
def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist=((1-corr)/2.)**.5 # distance matrix
    return dist
#———————————————————————————————————————
def getQuasiDiag(link):
    # Sort clustered items by distance
    link=link.astype(int) #Get Linkage Matrix 𝑌=〖{(𝑦_(𝑚,1),𝑦_(𝑚,2),𝑦_(𝑚,3),𝑦_(𝑚,4))}〗_(𝑚=1,…, 𝑁−1)
    sortIx=pd.Series([link[-1,0],link[-1,1]]) #Take out of last row of Linkage Matrix the first two columns
    numItems=link[-1,3] # number of original items e.g. SMI 20 Constituents
    while sortIx.max()>=numItems: #rows and columns of covariance matrix such that largest values lie along the diagonal
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space ; Info: range(Start, End, Steps)
        df0=sortIx[sortIx>=numItems] # find clusters
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1],index=i+1)
        sortIx=sortIx.append(df0) # item 2
        sortIx=sortIx.sort_index() # re-sort
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()
#———————————————————————————————————————
def plotCorrMatrix(path,corr,labels=None):
    # Heatmap of the correlation matrix
    if labels is None:labels=[]
    mpl.figure(num=None, figsize=(20, 10), dpi=600)
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.xticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.savefig(path)
    mpl.clf();mpl.close() # reset pylab
    return
#———————————————————————————————————————
def getRecBipart(cov,sortIx):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,len(i)/2), \
             (len(i)/2,len(i))) if len(i)>1] # bi-section
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
    return w
#———————————————————————————————————————
def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar
#———————————————————————————————————————
def getIVP(cov,**kargs):
    # Compute the inverse-variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp

#———————————————————————————————————————
#def generateData(nObs,size0,size1,sigma1):
#    # Time series of correlated variables
#    #1) generating some uncorrelated data
#   np.random.seed(seed=12345);random.seed(12345)
#   x=np.random.normal(0,1,size=(nObs,size0)) # each row is a variable
#    #2) creating correlation between the variables
#   cols=[random.randint(0,size0-1) for i in range(size1)]
#    y=x[:,cols]+np.random.normal(0,sigma1,size=(nObs,len(cols)))
#   x=np.append(x,y,axis=1)
#   x=pd.DataFrame(x,columns=range(1,x.shape[1]+1))
#   return x,cols

#———————————————————————————————————————
def main(path, todrop):
    #1) Import Data
    prices = pd.read_csv(path, index_col=0)
    prices = prices.drop(todrop,1)
    returns = (prices / prices.shift(1))-1
    returns = returns.drop(returns.index[range(0,3)])
    x = returns
      
    cov,corr=x.cov(),x.corr()

    #2) Cluster Data
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    mpl.figure(num=None, figsize=(20, 10), dpi=300, facecolor='w', edgecolor='k')    
    dn = sch.dendrogram(link, labels = dist.columns)
    mpl.savefig(general_path+"/Dendrogram.png")
    mpl.clf();mpl.close() # reset pylab
    
    #3) Diagonalize Data
    sortIx=getQuasiDiag(link)
    sortIx=corr.index[sortIx].tolist() # recover labels
    df0=corr.loc[sortIx,sortIx] # reorder
    
    #4)Create Correlation Matrix
    plotCorrMatrix(general_path+'/Correlation_Matrix.png',df0,labels=df0.columns)
    
    #5) Capital allocation
    hrp=getRecBipart(cov,sortIx)
    print(hrp)
    return hrp

#———————————————————————————————————————
if __name__=='__main__':main()

weights = main(general_path+"SMI.csv",".SSMI")

######Backtesting with synthetic data

def generateDataBacktest(nObs,sLength,size0,size1,mu0,sigma0,sigma1F):
    # Time series of correlated variables
    #1) generate random uncorrelated data
    x=np.random.normal(mu0,sigma0,size=(nObs,size0))
    #2) create correlation between the variables
    cols=[random.randint(0,size0-1) for i in xrange(size1)]
    y=x[:,cols]+np.random.normal(0,sigma0*sigma1F,size=(nObs,len(cols)))
    x=np.append(x,y,axis=1)
    #3) add common random shock
    point=np.random.randint(sLength,nObs-1,size=2)
    x[np.ix_(point,[cols[0],size0])]=np.array([[-.5,-.5],[2,2]])
    #4) add specific random shock
    point=np.random.randint(sLength,nObs-1,size=2)
    x[point,cols[-1]]=np.array([-.5,2])
    return x,cols
#———————————————————————————————————————
def getHRP(cov,corr):
    # Construct a hierarchical portfolio
    corr,cov=pd.DataFrame(corr),pd.DataFrame(cov)
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=getQuasiDiag(link)
    sortIx=corr.index[sortIx].tolist() # recover labels
    hrp=getRecBipart(cov,sortIx)
    return hrp.sort_index()
#———————————————————————————————————————
def getCLA(cov,**kargs):
# Compute CLA's minimum variance portfolio
    mean=np.arange(cov.shape[0]).reshape(-1,1) # Not used by C portf
    lB=np.zeros(mean.shape)
    uB=np.ones(mean.shape)
    cla=CLA.CLA(mean,cov,lB,uB)
    cla.solve()
    return cla.w[-1].flatten()
#———————————————————————————————————————
def hrpMC(numIters=1e4,nObs=520,size0=5,size1=5,mu0=0,sigma0=1e-2, \
    sigma1F=.25,sLength=260,rebal=22):
    # Monte Carlo experiment on HRP
    methods=[getIVP,getHRP,getCLA]
    stats,numIter={i.__name__:pd.Series() for i in methods},0 #Kreiere ein Dictionary der Form: Methodenname : Portfolio-Wert
    pointers=range(sLength,nObs,rebal) #Wir trainieren bis zur Sample Length und rebalancen dann alle rebal Tage im Testing-Set
    while numIter<numIters:
        print numIter
        #1) Prepare data for one experiment
        x,cols=generateDataBacktest(nObs,sLength,size0,size1,mu0,sigma0,sigma1F)
        r={i.__name__:pd.Series() for i in methods}
        #2) Compute portfolios in-sample
        for pointer in pointers:
            x_=x[pointer-sLength:pointer] #Zu jedem Rebalancing Zeitpunkt trainieren wir auf Basis der letzten sLength Resultate
            cov_,corr_=np.cov(x_,rowvar=0),np.corrcoef(x_,rowvar=0) #Autokovarianz und Autokorrelation
            #3) Compute performance out-of-sample
            x_=x[pointer:pointer+rebal] #Das hier ist unser Test-Set von diesem Rebalancing bis zum nächsten Rebalancing
            for func in methods:
                w_=func(cov=cov_,corr=corr_) # callback #Hier berechnen wir auf Basis der Varianz-Kovarianz-Matrix sowie der Korrelation die Gewichte mit jeder der drei Methoden
                r_=pd.Series(np.dot(x_,w_)) #Das Dot-Product von Gewichten mit den Assetpreisen zum Ende der Periode gibt uns den Portfoliowert zum Ende der Periode
                r[func.__name__]=r[func.__name__].append(r_) #Diesen Portfoliowert fügen wir an die Zeitreihe an
        #4) Evaluate and store results
        for func in methods:
            r_=r[func.__name__].reset_index(drop=True)
            p_=(1+r_).cumprod()
            stats[func.__name__].loc[numIter]=p_.iloc[-1]-1
        numIter+=1
    #5) Report results
    stats=pd.DataFrame.from_dict(stats,orient='columns')
    stats.to_csv('stats.csv')
    df0,df1,df2=stats.std(),stats.var(), stats.mean(axis=0)
    df3 = df2 / df0
    print pd.concat([df0,df1,df1/df1['getHRP']-1, df2, df3],axis=1)
    return

#0 = Standard deviation, 1 = Varianzen, 2 = Varianzen relativ zu HRP
