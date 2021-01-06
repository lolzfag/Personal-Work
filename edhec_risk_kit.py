import pandas as pd
'''
Tool kit we are developing ourselves
'''
def drawdown(return_series:pd.Series):
    '''
    Take a time series of asset returns,
    retuns a Dataframe with columns for
    the wealth index,
    the previous peaks,
    and the percentage drawdown
    '''
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({'Wealth': wealth_index,
                         'Previous Peak' : previous_peaks,
                         'Drawdown' : drawdowns        })

def get_ffme_returns():
    '''
    Load the Fama-French Dataset for returns of Top and Bottom Decile by MarketCap
    '''
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv',
                      header = 0, index_col = 0,na_values= -99.99)
    rets = me_m[['Lo 10','Hi 10']]
    rets.columns = ['SmallCap','LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format = '%Y%m').to_period('M')
    return rets

def get_hfi_returns():
    '''
    Load and format the EDHEC Hedge Fund Index Returns
    '''
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv',
                      header = 0, index_col = 0,parse_dates = True) # no need for renaming dates as the dates in the file are quite good formated
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def semideviation(r):
    '''
    Returns the semideviation aka negative semideviation of r, r must be a series/DF
    '''
    is_negative = r<0
    return r[is_negative].std(ddof=0)
    
def skewness(r):
    '''
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or Dataframe
    Returns a float or a Series
    '''
    demeaned_r = r - r.mean()
    # we use the population standard deviation which is different to python's calculation method(use N-1 not N, N is used in population S.D.), so set d.o.f(degrees of freedom)=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r **3

def kurtosis(r):
    '''
    Alternative to scipy.stats.kurtosis()
    It is worth to note that in scipy package, .kurtosis() gives excess kurtosis above 3
    Computes the kurtosis of the supplied Series or Dataframe
    Returns a float or a Series
    '''
    demeaned_r = r - r.mean()
    # we use the population standard deviation which is different to python's calculation method(use N-1 not N, N is used in population S.D.), so set d.o.f(degrees of freedom)=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r **4

import scipy.stats

def is_normal(r,level=0.01):
    '''
    Applies the jarque_Bera test to determine if a Series is normal or not, Test is applied at the 1% level
    Returns true if hypothesis of normality is accepted, False otherwise
    '''
    
    statistics, p_value = scipy.stats.jarque_bera(r)
    return p_value > level
    
import numpy as np

def var_historic(r, level=5):
    '''
    Var Historic(default 5% level indicates the possible loss at the worst 5% mark)
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level) # to return positive numbers due to convention
    else:
        raise TypeError('Expected r to be Series or DataFrame')
        
from scipy.stats import norm

def var_gaussian(r,level=5,modified=False):
    '''
    Returns the Parametric Gaussian VaR of a Series or DataFrame 
    default 5% level indicates the possible loss at worst 5% mark on the Gaussian Distribution Graph
    '''
    
    #compute the Z score assuming it was Gaussian
    z=norm.ppf(level/100)
    if modified:
        #modify the Z score based on observed skewness and kurtosis using Corner-fisher formula
        s = skewness(r)
        k = kurtosis(r)
        z =(z+
            (z**2 -1)*s/6 +
            (z**3 - 3*z)*(k-3)/24 -
            (2*z**3 - 5*z)*(s**2)/36
           )
    return -(r.mean()+z*r.std(ddof=0))

def cvar_historic(r,level = 5):
    '''
    Computer CVaR of Series/DataFrame
    '''

    if isinstance(r, pd.Series):
        is_beyond = r<=-var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError('Expected r to be a Series/DataFrame')