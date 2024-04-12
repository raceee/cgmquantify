import pandas as pd
import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

"""
        cgmquantify package
        Description:
        The cgmquantify package is a comprehensive library for computing metrics from continuous glucose monitors.

        Requirements:
        pandas, datetime, numpy, matplotlib, statsmodels

        Functions:
        importdexcom(): Imports data from Dexcom continuous glucose monitor devices
        interdaycv(): Computes and returns the interday coefficient of variation of glucose
        interdaysd(): Computes and returns the interday standard deviation of glucose
        intradaycv(): Computes and returns the intraday coefficient of variation of glucose 
        intradaysd(): Computes and returns the intraday standard deviation of glucose 
        TIR(): Computes and returns the time in range
        TOR(): Computes and returns the time outside range
        PIR(): Computes and returns the percent time in range
        POR(): Computes and returns the percent time outside range
        MGE(): Computes and returns the mean of glucose outside specified range
        MGN(): Computes and returns the mean of glucose inside specified range
        MAGE(): Computes and returns the mean amplitude of glucose excursions
        J_index(): Computes and returns the J-index
        LBGI(): Computes and returns the low blood glucose index
        HBGI(): Computes and returns the high blood glucose index
        ADRR(): Computes and returns the average daily risk range, an assessment of total daily glucose variations within risk space
        MODD(): Computes and returns the mean of daily differences. Examines mean of value + value 24 hours before
        CONGA24(): Computes and returns the continuous overall net glycemic action over 24 hours
        GMI(): Computes and returns the glucose management index
        eA1c(): Computes and returns the American Diabetes Association estimated HbA1c
        summary(): Computes and returns glucose summary metrics, including interday mean glucose, interday median glucose, interday minimum glucose, interday maximum glucose, interday first quartile glucose, and interday third quartile glucose
        plotglucosesd(): Plots glucose with specified standard deviation lines
        plotglucosebounds(): Plots glucose with user-defined boundaries
        plotglucosesmooth(): Plots smoothed glucose plot (with LOWESS smoothing)
                
"""

class CGMQuantify:
    def __init__(self, filename) -> None:
        self.df = self.importdexcom(filename)

    def importdexcom(self, filename):
        """
            Imports data from Dexcom continuous glucose monitor devices
            Args:
                filename (String): path to file
            Returns:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        """
        data = pd.read_csv(filename) 
        df = pd.DataFrame()
        df['Time'] = data['Timestamp (YYYY-MM-DDThh:mm:ss)']
        df['Glucose'] = pd.to_numeric(data['Glucose Value (mg/dL)'])
        df.drop(df.index[:12], inplace=True)
        df['Time'] =  pd.to_datetime(df['Time'], format='%Y-%m-%dT%H:%M:%S')
        df['Day'] = df['Time'].dt.date
        df = df.reset_index()
        return df


    def interdaycv(self):
        """
            Computes and returns the interday coefficient of variation of glucose
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                cvx (float): interday coefficient of variation averaged over all days
                
        """
        cvx = (np.std(self.df['Glucose']) / (np.mean(self.df['Glucose'])))*100
        return cvx

    def interdaysd(df):
        """
            Computes and returns the interday standard deviation of glucose
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                interdaysd (float): interday standard deviation averaged over all days
                
        """
        interdaysd = np.std(df['Glucose'])
        return interdaysd

    def intradaycv(self):
        """
            Computes and returns the intraday coefficient of variation of glucose 
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                intradaycv_mean (float): intraday coefficient of variation averaged over all days
                intradaycv_medan (float): intraday coefficient of variation median over all days
                intradaycv_sd (float): intraday coefficient of variation standard deviation over all days
                
        """
        intradaycv = []
        for i in pd.unique(self.df['Day']):
            intradaycv.append(self.interdaycv(self.df[self.df['Day']==i]))
        
        intradaycv_mean = np.mean(intradaycv)
        intradaycv_median = np.median(intradaycv)
        intradaycv_sd = np.std(intradaycv)
        
        return intradaycv_mean, intradaycv_median, intradaycv_sd


    def intradaysd(self):
        """
            Computes and returns the intraday standard deviation of glucose 
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                intradaysd_mean (float): intraday standard deviation averaged over all days
                intradaysd_medan (float): intraday standard deviation median over all days
                intradaysd_sd (float): intraday standard deviation standard deviation over all days
                
        """
        intradaysd =[]

        for i in pd.unique(self.df['Day']):
            intradaysd.append(np.std(self.df[self.df['Day']==i]))
        
        intradaysd_mean = np.mean(intradaysd)
        intradaysd_median = np.median(intradaysd)
        intradaysd_sd = np.std(intradaysd)
        return intradaysd_mean, intradaysd_median, intradaysd_sd

    def TIR(self, sd=1, sr=5):
        """
            Computes and returns the time in range
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                sd (integer): standard deviation for computing range (default=1)
                sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
            Returns:
                TIR (float): time in range, units=minutes
                
        """
        up = np.mean(self.df['Glucose']) + sd*np.std(self.df['Glucose'])
        dw = np.mean(self.df['Glucose']) - sd*np.std(self.df['Glucose'])
        TIR = len(self.df[(self.df['Glucose']<= up) & (self.df['Glucose']>= dw)])*sr 
        return TIR

    def TOR(self, sd=1, sr=5):
        """
            Computes and returns the time outside range
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                sd (integer): standard deviation for computing  range (default=1)
                sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
            Returns:
                TOR (float): time outside range, units=minutes
                
        """
        up = np.mean(self.df['Glucose']) + sd*np.std(self.df['Glucose'])
        dw = np.mean(self.df['Glucose']) - sd*np.std(self.df['Glucose'])
        TOR = len(self.df[(self.df['Glucose']>= up) | (self.df['Glucose']<= dw)])*sr
        return TOR

    def POR(self, sd=1, sr=5):
        """
            Computes and returns the percent time outside range
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                sd (integer): standard deviation for computing range (default=1)
                sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
            Returns:
                POR (float): percent time outside range, units=%
                
        """
        up = np.mean(self.df['Glucose']) + sd*np.std(self.df['Glucose'])
        dw = np.mean(self.df['Glucose']) - sd*np.std(self.df['Glucose'])
        TOR = len(self.df[(self.df['Glucose']>= up) | (self.df['Glucose']<= dw)])*sr
        POR = (TOR/(len(self.df)*sr))*100
        return POR

    def PIR(self, sd=1, sr=5):
        """
            Computes and returns the percent time inside range
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                sd (integer): standard deviation for computing range (default=1)
                sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
            Returns:
                PIR (float): percent time inside range, units=%
                
        """
        up = np.mean(self.df['Glucose']) + sd*np.std(self.df['Glucose'])
        dw = np.mean(self.df['Glucose']) - sd*np.std(self.df['Glucose'])
        TIR = len(self.df[(self.df['Glucose']<= up) | (self.df['Glucose']>= dw)])*sr
        PIR = (TIR/(len(self.df)*sr))*100
        return PIR

    def MGE(self, sd=1):
        """
            Computes and returns the mean of glucose outside specified range
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                sd (integer): standard deviation for computing range (default=1)
            Returns:
                MGE (float): the mean of glucose excursions (outside specified range)
                
        """
        up = np.mean(self.df['Glucose']) + sd*np.std(self.df['Glucose'])
        dw = np.mean(self.df['Glucose']) - sd*np.std(self.df['Glucose'])
        MGE = np.mean(self.df[(self.df['Glucose']>= up) | (self.df['Glucose']<= dw)])
        return MGE

    def MGN(self, sd=1):
        """
            Computes and returns the mean of glucose inside specified range
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                sd (integer): standard deviation for computing range (default=1)
            Returns:
                MGN (float): the mean of glucose excursions (inside specified range)
                
        """
        up = np.mean(self.df['Glucose']) + sd*np.std(self.df['Glucose'])
        dw = np.mean(self.df['Glucose']) - sd*np.std(self.df['Glucose'])
        MGN = np.mean(self.df[(self.df['Glucose']<= up) & (self.df['Glucose']>= dw)])
        return MGN

    def MAGE(self, std=1):
        """
            Computes and returns the mean amplitude of glucose excursions
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                sd (integer): standard deviation for computing range (default=1)
            Returns:
                MAGE (float): the mean amplitude of glucose excursions 
            Refs:
                Sneh Gajiwala: https://github.com/snehG0205/NCSA_genomics/tree/2bfbb87c9c872b1458ef3597d9fb2e56ac13ad64
                
        """
            
            #extracting glucose values and incdices
            glucose = self.df['Glucose'].tolist()
            ix = [1*i for i in range(len(glucose))]
            stdev = std
            
            # local minima & maxima
            a = np.diff(np.sign(np.diff(glucose))).nonzero()[0] + 1      
            # local min
            valleys = (np.diff(np.sign(np.diff(glucose))) > 0).nonzero()[0] + 1 
            # local max
            peaks = (np.diff(np.sign(np.diff(glucose))) < 0).nonzero()[0] + 1         
            # +1 -- diff reduces original index number

            #store local minima and maxima -> identify + remove turning points
            excursion_points = pd.DataFrame(columns=['Index', 'Time', 'Glucose', 'Type'])
            k=0
            for i in range(len(peaks)):
                excursion_points.loc[k] = [peaks[i]] + [self.df['Time'][k]] + [self.df['Glucose'][k]] + ["P"]
                k+=1

            for i in range(len(valleys)):
                excursion_points.loc[k] = [valleys[i]] + [self.df['Time'][k]] + [self.df['Glucose'][k]] + ["V"]
                k+=1

            excursion_points = excursion_points.sort_values(by=['Index'])
            excursion_points = excursion_points.reset_index(drop=True)


            # selecting turning points
            turning_points = pd.DataFrame(columns=['Index', 'Time', 'Glucose', 'Type'])
            k=0
            for i in range(stdev,len(excursion_points.Index)-stdev):
                positions = [i-stdev,i,i+stdev]
                for j in range(0,len(positions)-1):
                    if(excursion_points.Type[positions[j]] == excursion_points.Type[positions[j+1]]):
                        if(excursion_points.Type[positions[j]]=='P'):
                            if excursion_points.Glucose[positions[j]]>=excursion_points.Glucose[positions[j+1]]:
                                turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                                k+=1
                            else:
                                turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                                k+=1
                        else:
                            if excursion_points.Glucose[positions[j]]<=excursion_points.Glucose[positions[j+1]]:
                                turning_points.loc[k] = excursion_points.loc[positions[j]]
                                k+=1
                            else:
                                turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                                k+=1

            if len(turning_points.index)<10:
                turning_points = excursion_points.copy()
                excursion_count = len(excursion_points.index)
            else:
                excursion_count = len(excursion_points.index)/2


            turning_points = turning_points.drop_duplicates(subset= "Index", keep= "first")
            turning_points=turning_points.reset_index(drop=True)
            excursion_points = excursion_points[excursion_points.Index.isin(turning_points.Index) == False]
            excursion_points = excursion_points.reset_index(drop=True)

            # calculating MAGE
            mage = turning_points.Glucose.sum()/excursion_count
            
            return round(mage,3)



    def J_index(self):
        """
            Computes and returns the J-index
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                J (float): J-index of glucose
                
        """
        J = 0.001*((np.mean(self.df['Glucose'])+np.std(self.df['Glucose']))**2)
        return J

    def LBGI_HBGI(self):
        """
            Connecter function to calculate rh and rl, used for ADRR function
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                LBGI (float): Low blood glucose index
                HBGI (float): High blood glucose index
                rl (float): See calculation of LBGI
                rh (float): See calculation of HBGI
                
        """
        f = ((np.log(self.df['Glucose'])**1.084) - 5.381)
        rl = []
        for i in f: 
            if (i <= 0):
                rl.append(22.77*(i**2))
            else:
                rl.append(0)

        LBGI = np.mean(rl)

        rh = []
        for i in f: 
            if (i > 0):
                rh.append(22.77*(i**2))
            else:
                rh.append(0)

        HBGI = np.mean(rh)
        
        return LBGI, HBGI, rh, rl



    def LBGI(self):
        """
            Computes and returns the low blood glucose index
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                LBGI (float): Low blood glucose index
                
        """
        f = ((np.log(self.df['Glucose'])**1.084) - 5.381)
        rl = []
        for i in f: 
            if (i <= 0):
                rl.append(22.77*(i**2))
            else:
                rl.append(0)

        LBGI = np.mean(rl)
        return LBGI

    def HBGI(self):
        """
            Computes and returns the high blood glucose index
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                HBGI (float): High blood glucose index
                
        """
        f = ((np.log(self.df['Glucose'])**1.084) - 5.381)
        rh = []
        for i in f: 
            if (i > 0):
                rh.append(22.77*(i**2))
            else:
                rh.append(0)

        HBGI = np.mean(rh)
        return HBGI

    def ADRR(self):
        """
            Computes and returns the average daily risk range, an assessment of total daily glucose variations within risk space
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                ADRRx (float): average daily risk range
                
        """
        ADRRl = []
        for i in pd.unique(self.df['Day']):
            LBGI, HBGI, rh, rl = LBGI_HBGI(self.df[self.df['Day']==i])
            LR = np.max(rl)
            HR = np.max(rh)
            ADRRl.append(LR+HR)

        ADRRx = np.mean(ADRRl)
        return ADRRx

    def uniquevalfilter(self, value):
        """
            Supporting function for MODD and CONGA24 functions
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                value (datetime): time to match up with previous 24 hours
            Returns:
                MODD_n (float): Best matched with unique value, value
                
        """
        xdf = self.df[self.df['Minfrommid'] == value]
        n = len(xdf)
        diff = abs(xdf['Glucose'].diff())
        MODD_n = np.nanmean(diff)
        return MODD_n

    def MODD(self):
        """
            Computes and returns the mean of daily differences. Examines mean of value + value 24 hours before
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Requires:
                uniquevalfilter (function)
            Returns:
                MODD (float): Mean of daily differences
                
        """
        self.df['Timefrommidnight'] =  self.df['Time'].dt.time
        lists=[]
        for i in range(0, len(self.df['Timefrommidnight'])):
            lists.append(int(self.df['Timefrommidnight'][i].strftime('%H:%M:%S')[0:2])*60 + int(self.df['Timefrommidnight'][i].strftime('%H:%M:%S')[3:5]) + round(int(self.df['Timefrommidnight'][i].strftime('%H:%M:%S')[6:9])/60))
        self.df['Minfrommid'] = lists
        self.df = self.df.drop(columns=['Timefrommidnight'])
        
        #Calculation of MODD and CONGA:
        MODD_n = []
        uniquetimes = self.df['Minfrommid'].unique()

        for i in uniquetimes:
            MODD_n.append(uniquevalfilter(self.df, i))
        
        #Remove zeros from dataframe for calculation (in case there are random unique values that result in a mean of 0)
        MODD_n[MODD_n == 0] = np.nan
        
        MODD = np.nanmean(MODD_n)
        return MODD

    def CONGA24(self):
        """
            Computes and returns the continuous overall net glycemic action over 24 hours
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Requires:
                uniquevalfilter (function)
            Returns:
                CONGA24 (float): continuous overall net glycemic action over 24 hours
                
        """
        self.df['Timefrommidnight'] =  self.df['Time'].dt.time
        lists=[]
        for i in range(0, len(df['Timefrommidnight'])):
            lists.append(int(self.df['Timefrommidnight'][i].strftime('%H:%M:%S')[0:2])*60 + int(self.df['Timefrommidnight'][i].strftime('%H:%M:%S')[3:5]) + round(int(self.df['Timefrommidnight'][i].strftime('%H:%M:%S')[6:9])/60))
        self.df['Minfrommid'] = lists
        self.df = self.df.drop(columns=['Timefrommidnight'])
        
        #Calculation of MODD and CONGA:
        MODD_n = []
        uniquetimes = self.df['Minfrommid'].unique()

        for i in uniquetimes:
            MODD_n.append(uniquevalfilter(self.df, i))
        
        #Remove zeros from dataframe for calculation (in case there are random unique values that result in a mean of 0)
        MODD_n[MODD_n == 0] = np.nan
        
        CONGA24 = np.nanstd(MODD_n)
        return CONGA24

    def GMI(self):
        """
            Computes and returns the glucose management index
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                GMI (float): glucose management index (an estimate of HbA1c)
                
        """
        GMI = 3.31 + (0.02392*np.mean(self.df['Glucose']))
        return GMI

    def eA1c(self):
        """
            Computes and returns the American Diabetes Association estimated HbA1c
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                eA1c (float): an estimate of HbA1c from the American Diabetes Association
                
        """
        eA1c = (46.7 + np.mean(self.df['Glucose']))/ 28.7 
        return eA1c

    def summary(self): 
        """
            Computes and returns glucose summary metrics
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            Returns:
                meanG (float): interday mean of glucose
                medianG (float): interday median of glucose
                minG (float): interday minimum of glucose
                maxG (float): interday maximum of glucose
                Q1G (float): interday first quartile of glucose
                Q3G (float): interday third quartile of glucose
                
        """
        meanG = np.nanmean(self.df['Glucose'])
        medianG = np.nanmedian(self.df['Glucose'])
        minG = np.nanmin(self.df['Glucose'])
        maxG = np.nanmax(self.df['Glucose'])
        Q1G = np.nanpercentile(self.df['Glucose'], 25)
        Q3G = np.nanpercentile(self.df['Glucose'], 75)
        
        return meanG, medianG, minG, maxG, Q1G, Q3G

    def plotglucosesd(self, sd=1, size=15):
        """
            Plots glucose with specified standard deviation lines
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                sd (integer): standard deviation lines to plot (default=1)
                size (integer): font size for plot (default=15)
            Returns:
                plot of glucose with standard deviation lines
                
        """
        glucose_mean = np.mean(self.df['Glucose'])
        up = np.mean(self.df['Glucose']) + sd*np.std(self.df['Glucose'])
        dw = np.mean(self.df['Glucose']) - sd*np.std(self.df['Glucose'])

        plt.figure(figsize=(20,5))
        plt.rcParams.update({'font.size': size})
        plt.plot(self.df['Time'], self.df['Glucose'], '.', color = '#1f77b4')
        plt.axhline(y=glucose_mean, color='red', linestyle='-')
        plt.axhline(y=up, color='pink', linestyle='-')
        plt.axhline(y=dw, color='pink', linestyle='-')
        plt.ylabel('Glucose')
        plt.show()

    def plotglucosebounds(self, upperbound = 180, lowerbound = 70, size=15):
        """
            Plots glucose with user-defined boundaries
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                upperbound (integer): user defined upper bound for glucose line to plot (default=180)
                lowerbound (integer): user defined lower bound for glucose line to plot (default=70)
                size (integer): font size for plot (default=15)
            Returns:
                plot of glucose with user defined boundary lines
                
        """
        plt.figure(figsize=(20,5))
        plt.rcParams.update({'font.size': size})
        plt.plot(self.df['Time'], self.df['Glucose'], '.', color = '#1f77b4')
        plt.axhline(y=upperbound, color='red', linestyle='-')
        plt.axhline(y=lowerbound, color='orange', linestyle='-')
        plt.ylabel('Glucose')
        plt.show()

    def plotglucosesmooth(self, size=15):
        """
            Plots smoothed glucose plot (with LOWESS smoothing)
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                size (integer): font size for plot (default=15)
            Returns:
                LOWESS-smoothed plot of glucose
                
        """
        filteres = lowess(self.df['Glucose'], self.df['Time'], is_sorted=True, frac=0.025, it=0)
        filtered = pd.to_datetime(filteres[:,0], format='%Y-%m-%dT%H:%M:%S') 
        
        plt.figure(figsize=(20,5))
        plt.rcParams.update({'font.size': size})
        plt.plot(self.df['Time'], self.df['Glucose'], '.')
        plt.plot(filtered, filteres[:,1], 'r')
        plt.ylabel('Glucose')
        plt.show()
