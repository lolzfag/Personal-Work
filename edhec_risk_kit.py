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
    