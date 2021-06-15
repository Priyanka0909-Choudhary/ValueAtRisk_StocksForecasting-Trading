#!/usr/bin/env python
# coding: utf-8

# ## <center>VAR Using Monte Carlo Simulation (Random Walk Function-Geometric Brownian motion(GBM)) and Stock Forcasting</center>

# This notebook contains the expected loss margin ( daily, weekly, monthly, quarterly,  yearly) using Monte carlo simulation. This is calculated for Delta Coorporation, Sun tv, Idea, Alembic Pharma Limited, ITC, and United Spirits Limited selected after a fundamental analysis. The process uses the Historical Data extracted using the yfinance API. It also has Support and Resistance Levels mapped out from the Caculated VaR . It predicts the future prices of the stocks using the historical prices in conjuction with the Efficient Market Hypothesis.

# In[183]:


#Installing the reqired libraries
get_ipython().system('pip install yfinance')
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from scipy.stats import norm
import scipy.stats
import matplotlib.pyplot as plt


# In[184]:


#Getting  Historical data from yfinance
data=yf.download(tickers='DELTACORP.NS', period='5y', interval='1mo')
data.tail()


# In[46]:


data.shape


# In[209]:


fig = plt.figure()
fig.set_size_inches(10,3)
data["Adj Close"].pct_change().plot()
plt.title("Delta Corp monthly returns", weight="bold");
plt.grid()


# In[190]:


returns=data["Adj Close"].pct_change().dropna()
mean = returns.mean()
sigmaa = returns.std()
x=returns.quantile(0.01)
x


# In[191]:


y=returns.quantile(0.05)
y


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A MONTH STARTING FROM 11/06/2021 FOR DELTA CORP

# In[192]:


# defining the time frame, drift, and the volatility
days = 22 # monthly
dt = 1/float(days)  #time horizon(t/n)
sigma = sigmaa # volatility
mu = mean  # drift (average growth rate)


# In[193]:


# function random_walk using the GBM Concept(Geometric Brownian Motion) for risk management. The breakdown for the change in price is such that it has a drift
#and a shock calculated using Historical data Mean and standard deviation for daily, monthly and quarterly returns on the selected stocks
def random_walk(startprice):
    price = np.zeros(days)
    shock = np.zeros(days)
    price[0] = startprice
    for i in range(1, days):
        shock[i] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        price[i] = max(0, price[i-1] + shock[i] * price[i-1])
    return price


# In[194]:


Run the random function for 1000 simulations
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(183.550003))             #Last Traded price for Delta Coorporation
# If I start from today onwards, the simulation is for next 22 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 22 DAYS", weight="bold");
# The exceptation is more towards the profit side owing to the positive drift(mean) calculated from monthly historical data


# In[195]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(183.550003)[days-1]
q = np.percentile(simulations, 1)     #at 99% confidence
t = np.percentile(simulations,5)      #at 95% confidence
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:183.550003 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(183.550003 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(183.550003 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A WEEK STARTING FROM 11/06/2021 FOR DELTA CORP

# In[211]:


#Extracting weekly data
data1=yf.download(tickers='DELTACORP.NS', period='5y', interval='1wk')
data1.tail()


# In[212]:


fig1 = plt.figure()
fig1.set_size_inches(10,3)
data1["Adj Close"].pct_change().plot()
plt.title("Delta Corp weekly returns", weight="bold");
plt.grid()


# In[198]:


returns1=data1["Adj Close"].pct_change().dropna()
mean1 = returns1.mean()
sigma1 = returns1.std()
x1=returns1.quantile(0.05)
x1


# In[199]:


y1=returns1.quantile(0.01)
y1


# In[200]:


# Monte carlo using the random function with 1000 iterations
days = 5 # weekly
dt = 1/float(days)
sigma = sigma1
mu = mean1
# function random_walk to define the random path starting from (0,0)
def random_walk(startprice):
    price = np.zeros(days)
    shock = np.zeros(days)
    price[0] = startprice
    for i in range(1, days):
        shock[i] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        price[i] = max(0, price[i-1] + shock[i] * price[i-1])
    return price
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(183.550003))   # Last Traded Price
# If I start from today onwards, the simulation is for next 5 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 5 DAYS", weight="bold");
# The exceptation is more towards the profit side owing to the positive drift(mean outcome)


# In[201]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(183.550003)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:183.550003 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(183.550003 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(183.550003 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A QUARTER STARTING FROM 11/06/2021 FOR DELTA CORP

# In[259]:


#Extracting weekly data
data12=yf.download(tickers='DELTACORP.NS', period='5y', interval='3mo')
data12.tail()


# In[260]:


fig12 = plt.figure()
fig12.set_size_inches(10,3)
data12["Adj Close"].pct_change().plot()
plt.title("Delta Corp quarterly returns", weight="bold");
plt.grid()


# In[261]:


returns12=data12["Adj Close"].pct_change().dropna()
mean12 = returns12.mean()
sigma12 = returns12.std()
x12=returns12.quantile(0.05)
x12


# In[205]:


y12=returns12.quantile(0.01)
y12


# In[262]:


# Monte carlo using the random function with 10000 iterations
days = 66 # weekly
dt = 1/float(days)
sigma = sigma12
mu = mean12
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(183.550003))   # Last Traded Price
# If I start from today onwards, the simulation is for next 66 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER A QUARTER", weight="bold");
# The exceptation is more towards the profit side owing to the positive drift(mean=~0.0075) calculated from quaterly historical data


# In[264]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(183.550003)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:183.550003 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(183.550003 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(183.550003 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A YEAR STARTING FROM 11/06/2021 FOR DELTA CORP

# In[220]:


#Extracting data
data13=yf.download(tickers='DELTACORP.NS', period='1mo', interval='1d')
data13.tail()


# In[221]:


fig13 = plt.figure()
fig13.set_size_inches(10,3)
data13["Adj Close"].pct_change().plot()
plt.title("Delta Corp daily returns in last year", weight="bold");
plt.grid()


# In[222]:


returns13=data13["Adj Close"].pct_change().dropna()
mean13 = returns13.mean()
sigma13 = returns13.std()
x13=returns13.quantile(0.05)
x13
#With 95% confidence, worst daily loss won't exceed 1.8%


# In[223]:


y13=returns13.quantile(0.01)
y13
#With 99% confidence, worst daily loss won't exceed 2.05%


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A MONTH STARTING FROM 11/06/2021 FOR SUN TV

# In[265]:


#Getting  Historical data from yfinance
data2=yf.download(tickers='SUNTV.NS', period='5y', interval='1mo')
data2.tail()


# In[266]:


fig2 = plt.figure()
fig2.set_size_inches(10,3)
data2["Adj Close"].pct_change().plot()
plt.title("Sun tv monthly returns", weight="bold");
plt.grid()


# In[267]:


returns2=data2["Adj Close"].pct_change().dropna()
mean2 = returns2.mean()
sigma2 = returns2.std()
x2=returns2.quantile(0.05)
x2


# In[75]:


y2=returns2.quantile(0.01)
y2


# In[268]:


days = 22 # monthly
dt = 1/float(days)
sigma = sigma2 
mu = mean2  
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(531.049988))
# If I start from today onwards, the simulation is for next 22 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 22 DAYS", weight="bold");
# The exceptation is almost same on both side but slightly more towards the profit side owing to the positive drift(mean=~0.0022) calculated from monthly historical data


# In[272]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(531.049988)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:531.049988 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(531.049988 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(531.049988 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A WEEK STARTING FROM 10/06/2021 FOR SUN TV

# In[227]:


data3=yf.download(tickers='SUNTV.NS', period='5y', interval='1wk')
data3.tail()


# In[228]:


fig3 = plt.figure()
fig3.set_size_inches(10,3)
data3["Adj Close"].pct_change().plot()
plt.title("Sun tv daily weekly returns", weight="bold");
plt.grid()


# In[80]:


returns3=data3["Adj Close"].pct_change().dropna()
mean3 = returns3.mean()
sigma3 = returns3.std()
x3=returns3.quantile(0.05)
x3


# In[81]:


y3=returns3.quantile(0.01)
y3


# In[82]:


days = 5 # weekly
dt = 1/float(days)
sigma = sigma3
mu = mean3 
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(531.049988))   # last traded price
# If I start from today onwards, the simulation is for next 5 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 5 DAYS", weight="bold");
# The exceptation is more towards the loss side owing to the negative drift(mean=-0.002) calculated from weekly historical data


# In[83]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(531.049988)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:531.049988 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(531.049988 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(531.049988 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


#    ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A QUARTER STARTING FROM 10/06/2021 FOR SUN TV

# In[84]:


data20=yf.download(tickers='SUNTV.NS', period='5y', interval='3mo')
data20.tail()


# In[229]:


fig20 = plt.figure()
fig20.set_size_inches(10,3)
data20["Adj Close"].pct_change().plot()
plt.title("SUN TV daily quarterly returns", weight="bold");
plt.grid()


# In[86]:


returns20=data20["Adj Close"].pct_change().dropna()
mean20 = returns20.mean()
sigma20 = returns20.std()
x20=returns20.quantile(0.05)
x20


# In[87]:


y20=returns20.quantile(0.01)
y20


# In[90]:


# Monte carlo using the random function with 10000 iterations
days = 66 # weekly
dt = 1/float(days)
sigma = sigma20
mu = mean20
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(531.049988))   # Last Traded Price
# If I start from today onwards, the simulation is for next 252 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER A YEAR", weight="bold");
# The exceptation is more towards the profit side owing to the positive drift(mean=~0.0075) calculated from yearly historical data


# In[91]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(531.049988)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:531.049988 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(531.049988 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(531.049988 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A MONTH STARTING FROM 11/06/2021 FOR ITC

# In[231]:


#Getting  Historical data from yfinance
data4=yf.download(tickers='ITC.NS', period='5y', interval='1mo')
data4.tail()


# In[232]:


fig4 = plt.figure()
fig4.set_size_inches(10,3)
data4["Adj Close"].pct_change().plot()
plt.title("ITC daily monthly returns", weight="bold");
plt.grid()


# In[233]:


returns4=data4["Adj Close"].pct_change().dropna()
mean4 = returns4.mean()
sigma4 = returns4.std()
x4=returns4.quantile(0.05)
x4


# In[234]:


y4=returns4.quantile(0.01)
y4


# In[97]:


days = 22 # monthly
dt = 1/float(days)
sigma = sigma4 
mu = mean4  
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(207.899994))   # last traded price
# If I start from today onwards, the simulation is for next 22 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 22 DAYS", weight="bold");
# The exceptation is almost same on both side but slightly more towards the profit side owing to the positive drift(mean=~0.0016) calculated from monthly historical data


# In[98]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(207.899994)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:207.899994 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(207.899994 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(207.899994 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A WEEK STARTING FROM 11/06/2021 FOR ITC

# In[236]:


data5=yf.download(tickers='ITC.NS', period='5y', interval='1wk')
data5.tail()


# In[237]:


fig5 = plt.figure()
fig5.set_size_inches(10,3)
data5["Adj Close"].pct_change().plot()
plt.title("ITC daily weekly returns", weight="bold");
plt.grid()


# In[235]:


returns5=data5["Adj Close"].pct_change().dropna()
mean5 = returns5.mean()
sigma5 = returns5.std()
x5=returns5.quantile(0.05)
x5


# In[104]:


y5=returns5.quantile(0.01)
y5


# In[105]:


days = 5 # weekly
dt = 1/float(days)
sigma = sigma5
mu = mean5
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(207.899994))
# If I start from today onwards, the simulation is for next 5 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 5 DAYS", weight="bold");


# In[106]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(207.899994)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:207.899994 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(207.899994 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(207.899994 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A QUARTER STARTING FROM 11/06/2021 FOR ITC

# In[238]:


data21=yf.download(tickers='ITC.NS', period='5y', interval='3mo')
data21.tail()


# In[239]:


fig21 = plt.figure()
fig21.set_size_inches(10,3)
data21["Adj Close"].pct_change().plot()
plt.title("ITC daily quarterly returns", weight="bold");
plt.grid()


# In[240]:


returns21=data21["Adj Close"].pct_change().dropna()
mean21 = returns21.mean()
sigma21 = returns21.std()
x21=returns21.quantile(0.05)
x21


# In[241]:


y21=returns21.quantile(0.01)
y21


# In[112]:


# Monte carlo using the random function with 10000 iterations
days = 252 # weekly
dt = 1/float(days)
sigma = sigma21
mu = mean21
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(207.899994))   # Last Traded Price
# If I start from today onwards, the simulation is for next 252 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER A YEAR", weight="bold");
# The exceptation is more towards the profit side owing to the positive drift(mean=~0.0075) calculated from yearly historical data


# In[113]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(207.899994)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:207.899994 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(207.899994 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(207.899994 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A MONTH STARTING FROM 11/06/2021 FOR UNITED SPIRITS

# In[243]:


#Getting  Historical data from yfinance
data6=yf.download(tickers='MCDOWELL-N.NS', period='5y', interval='1mo')
data6.tail()


# In[244]:


fig6 = plt.figure()
fig6.set_size_inches(10,3)
data6["Adj Close"].pct_change().plot()
plt.title("UNITED SPIRIT LIMITED monthly returns", weight="bold");
plt.grid()


# In[116]:


returns6=data6["Adj Close"].pct_change().dropna()
mean6 = returns6.mean()
sigma6 = returns6.std()
x6=returns6.quantile(0.05)
x6


# In[117]:


y6=returns6.quantile(0.01)
y6


# In[118]:


days = 22 # monthly
dt = 1/float(days)
sigma = sigma6 
mu = mean6 
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(646.049988))
# If I start from today onwards, the simulation is for next 22 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 22 DAYS", weight="bold");
# The exceptation is almost same on both side but slightly more towards the profit side owing to the positive drift(mean=~0.006) calculated from monthly historical data


# In[119]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(646.049988)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:646.049988 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(646.049988 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(646.049988 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A WEEK STARTING FROM 10/06/2021 FOR UNITED SPIRITS

# In[121]:


data7=yf.download(tickers='MCDOWELL-N.NS', period='5y', interval='1wk')
data7.tail()


# In[245]:


fig7 = plt.figure()
fig7.set_size_inches(10,3)
data7["Adj Close"].pct_change().plot()
plt.title("USL returns in last week", weight="bold");
plt.grid()


# In[123]:


returns7=data7["Adj Close"].pct_change().dropna()
mean7 = returns7.mean()
sigma7 = returns7.std()
x7=returns7.quantile(0.05)
x7


# In[124]:


y7=returns7.quantile(0.01)
y7


# In[125]:


days = 5 # weekly
dt = 1/float(days)
sigma = sigma7
mu = mean7 
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(646.049988))
# If I start from today onwards, the simulation is for next 5 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 5 DAYS", weight="bold");
# The exceptation is more towards the profit owing to the positive drift(mean=0.005) calculated from weekly historical data


# In[126]:


runs = 10000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(646.049988)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:646.049988 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(646.049988 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(646.049988 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A QUARTER STARTING FROM 11/06/2021 FOR UNITED SPIRITS

# In[246]:


data22=yf.download(tickers='MCDOWELL-N.NS', period='5y', interval='3mo')
data22.tail()


# In[247]:


fig22 = plt.figure()
fig22.set_size_inches(10,3)
data22["Adj Close"].pct_change().plot()
plt.title("USL quarterly returns", weight="bold");
plt.grid()


# In[129]:


returns22=data22["Adj Close"].pct_change().dropna()
mean22 = returns22.mean()
sigma22 = returns22.std()
x22=returns22.quantile(0.05)
x22


# In[130]:


y22=returns22.quantile(0.01)
y22


# In[131]:


# Monte carlo using the random function with 10000 iterations
days = 252 # weekly
dt = 1/float(days)
sigma = sigma22
mu = mean22
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(646.049988))   # Last Traded Price
# If I start from today onwards, the simulation is for next 252 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER A YEAR", weight="bold");
# The exceptation is more towards the profit side owing to the positive drift(mean=~0.0075) calculated from yearly historical data


# In[132]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(646.049988)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:646.049988 ")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(646.049988 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(646.049988 - t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A MONTH STARTING FROM 11/06/2021 FOR ALEMBIC PHARMACEUTICALS

# In[133]:


#Getting  Historical data from yfinance
data8=yf.download(tickers='APLLTD.NS', period='5y', interval='1mo')
data8.tail()


# In[248]:


fig8 = plt.figure()
fig8.set_size_inches(10,3)
data8["Adj Close"].pct_change().plot()
plt.title("AP LIMITED monthly returns", weight="bold");
plt.grid()


# In[135]:


returns8=data8["Adj Close"].pct_change().dropna()
mean8 = returns8.mean()
sigma8 = returns8.std()
x8=returns8.quantile(0.05)
x8


# In[136]:


y8=returns8.quantile(0.01)
y8


# In[137]:


days = 22 # monthly
dt = 1/float(days)
sigma = sigma8 
mu = mean8  
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(1006.049988))
# If I start from today onwards, the simulation is for next 22 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 22 DAYS", weight="bold");
# The exceptation is almost same on both side but slightly more towards the profit side owing to the positive drift(mean) calculated from monthly historical data


# In[138]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(1006.049988)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:1006.049988")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(1006.049988 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(1006.049988- t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A WEEK STARTING FROM 11/06/2021 FOR AP LIMITED

# In[140]:


data9=yf.download(tickers='APLLTD.NS', period='5y', interval='1wk')
data9.tail()


# In[249]:


fig9 = plt.figure()
fig9.set_size_inches(10,3)
data9["Adj Close"].pct_change().plot()
plt.title("APLLTD weekly returns", weight="bold");
plt.grid()


# In[142]:


returns9=data9["Adj Close"].pct_change().dropna()
mean9 = returns9.mean()
sigma9 = returns9.std()
x9=returns9.quantile(0.05)
x9


# In[143]:


y9=returns9.quantile(0.01)
y9


# In[144]:


days = 5 # weekly
dt = 1/float(days)
sigma = sigma9
mu = mean9 
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(1006.049988))
# If I start from today onwards, the simulation is for next 5 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 5 DAYS", weight="bold");
# The exceptation is more towards the profit owing to the positive drift(mean=0.005) calculated from weekly historical data


# In[145]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(1006.049988)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:1006.049988")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(1006.049988 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(1006.049988- t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A QUARTER STARTING FROM 11/06/2021 FOR APLLTD

# In[146]:


data23=yf.download(tickers='APLLTD.NS', period='5y', interval='3mo')
data23.tail()


# In[250]:


fig23 = plt.figure()
fig23.set_size_inches(10,3)
data23["Adj Close"].pct_change().plot()
plt.title("APLLTD daily quarterly returns", weight="bold");
plt.grid()


# In[148]:


returns23=data23["Adj Close"].pct_change().dropna()
mean23 = returns23.mean()
sigma23 = returns23.std()
x23=returns23.quantile(0.05)
x23


# In[149]:


y23=returns23.quantile(0.01)
y23


# In[152]:


# Monte carlo using the random function with 1000 iterations
days = 66 # weekly
dt = 1/float(days)
sigma = sigma23
mu = mean23
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(1006.04998))   # Last Traded Price
# If I start from today onwards, the simulation is for next 252 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER A QUARTER", weight="bold");
# The exceptation is more towards the profit side owing to the positive drift(mean=~0.0075) calculated from yearly historical data


# In[153]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(1006.049988)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:1006.049988")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(1006.049988 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(1006.049988- t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# In[ ]:





# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A MONTH STARTING FROM 11/06/2021 FOR IDEA

# In[251]:


#Getting  Historical data from yfinance
data10=yf.download(tickers='IDEA.NS', period='5y', interval='1mo')
data10.tail()


# In[252]:


fig10 = plt.figure()
fig10.set_size_inches(10,3)
data10["Adj Close"].pct_change().plot()
plt.title("IDEA daily monthly returns", weight="bold");
plt.grid()


# In[157]:


returns10=data10["Adj Close"].pct_change().dropna()
mean10 = returns10.mean()
sigma10 = returns10.std()
x10=returns10.quantile(0.05)
x10


# In[158]:


y10=returns10.quantile(0.01)
y10


# In[159]:


days = 22 # monthly
dt = 1/float(days)
sigma = sigma10 
mu = mean10 
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(9.85))
# If I start from today onwards, the simulation is for next 22 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 22 DAYS", weight="bold");
# The exceptation is almost same on both side but slightly more towards the profit side owing to the positive drift(mean) calculated from monthly historical data


# In[160]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(9.85)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:9.85")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(9.85 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(9.85- t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A WEEK STARTING FROM 11/06/2021 FOR IDEA

# In[162]:


data11=yf.download(tickers='IDEA.NS', period='5y', interval='1wk')
data11.tail()


# In[254]:


fig11 = plt.figure()
fig11.set_size_inches(10,3)
data11["Adj Close"].pct_change().plot()
plt.title("IDEA weekly returns", weight="bold");
plt.grid()


# In[255]:


returns11=data11["Adj Close"].pct_change().dropna()
mean11 = returns11.mean()
sigma11 = returns11.std()
x11=returns11.quantile(0.05)
x11


# In[165]:


y11=returns11.quantile(0.01)
y11


# In[166]:


days = 5 # weekly
dt = 1/float(days)
sigma = sigma11
mu = mean11 
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(9.85))
# If I start from today onwards, the simulation is for next 5 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER 5 DAYS", weight="bold");
# The exceptation is more towards the profit owing to the positive drift(mean=0.005) calculated from weekly historical data


# In[167]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(9.85)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:9.85")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(9.85 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(9.85- t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# ### MONTE CARLO SIMULATION TO CALCULATE THE EXPECTED LOSS FOR A QUARTERLY STARTING FROM 11/06/2021 FOR IDEA

# In[256]:


data24=yf.download(tickers='IDEA.NS', period='5y', interval='3mo')
data24.tail()


# In[258]:


fig24 = plt.figure()
fig24.set_size_inches(10,3)
data24["Adj Close"].pct_change().plot()
plt.title("IDEA quarterly returns", weight="bold");
plt.grid()


# In[177]:


returns24=data24["Adj Close"].pct_change().dropna()
mean24 = returns24.mean()
sigma24 = returns24.std()
x24=returns24.quantile(0.05)
x24


# In[178]:


y24=returns24.quantile(0.01)
y24


# In[181]:


# Monte carlo using the random function with 10000 iterations
days = 66 # weekly
dt = 1/float(days)
sigma = sigma24
mu = mean24
plt.figure(figsize=(9,4))    
for run in range(1000):
    plt.plot(random_walk(9.85))   # Last Traded Price
# If I start from today onwards, the simulation is for next 252 days
plt.xlabel("Time_days")
plt.ylabel("Price");
plt.grid()
plt.title("EXPECTED PRICE RANGE AFTER A YEAR", weight="bold");
# The exceptation is more towards the profit side owing to the positive drift calculated from yearly historical data


# In[182]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(9.85)[days-1]
q = np.percentile(simulations, 1)
t = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
plt.figtext(0.6, 0.8, "Start price:9.85")
plt.figtext(0.6, 0.7, "Mean final price: {:.3}".format(simulations.mean()))
plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}".format(9.85 - q))
plt.figtext(0.6, 0.5, "VaR(0.95): {:.3}".format(9.85- t))
plt.title("Final price distribution after {} days".format(days), weight="bold");


# In[276]:


data100=yf.download(tickers='IDEA.NS', period='2y', interval='3mo')
data100


# <center>
# <html>
#   <head>
#     <style>
#       table,
#       th,
#       td {
#         padding: 10px;
#         border: 1px solid black;
#         border-collapse: collapse;
#       }
#     </style>
#   </head>
#   <body>
#       <table style="width:100%">
#           <tr>
#           <th>TICKER</th>
#           <th colspan="2">DAILY VAR(%)(95%,99%)</th>
#           <th colspan="2">D Support Levels</th>
#           <th colspan="2">D Resistance Levels</th>
#           <th colspan="2">WEEKLY VAR(%)</th>
#           <th colspan="2">WSL</th>
#           <th colspan="2">WRL</th>
#           <th colspan="2">MONTHLY VAR(%)</th>
#           <th colspan="2">MSL</th>
#           <th colspan="2">MRL</th>
#           <th colspan="2">QUARTERLY VAR(%)</th>
#           <th colspan="2">QSL</th>
#           <th colspan="2">QRL</th>
#           </tr>
#           <tr>
#           <td><b>DELTA CORP</b></td>
#           <td>2.09</td>
#           <td>1.99</td>
#           <td>180</td>
#           <td>180.19</td>
#           <td>187.69</td>
#           <td>187.51</td>
#           <td>14.87</td>
#           <td>10.08</td>
#           <td>156.25</td>
#           <td>165.05</td>
#           <td>210.85</td>
#           <td>202.05</td>
#           <td>28.82</td>
#           <td>22.23</td>
#           <td>130.65</td>
#           <td>142.75</td>
#           <td>236.45</td>
#           <td>224.35</td>
#           <td>35.9</td>
#           <td>28.8</td>
#           <td>117.65</td>
#           <td>130.95</td>
#           <td>249.45</td>
#           <td>236.15</td>
#           </tr>
#           <tr>
#           <td><b>SUN TV</b></td>
#           <td>1.67</td>
#           <td>1.84</td>
#           <td>522.58</td>
#           <td>521.67</td>
#           <td>540.32</td>
#           <td>541.23</td>
#           <td>10.62</td>
#           <td>7.83</td>
#           <td>474.65</td>
#           <td>489.45</td>
#           <td>587.45</td>
#           <td>572.65</td>
#           <td>20.15</td>
#           <td>15.4</td>
#           <td>424.05</td>
#           <td>449.25</td>
#           <td>638.05</td>
#           <td>612.85</td>
#           <td>26.55</td>
#           <td>17.87</td>
#           <td>390.05</td>
#           <td>436.15</td>
#           <td>672.05</td>
#           <td>625.95</td>
#           </tr>
#           <tr>
#           <td><b>ITC</b></td>
#           <td>0.44</td>
#           <td>0.5</td>
#           <td>208.38</td>
#           <td>208.25</td>
#           <td>210.22</td>
#           <td>210.35</td>
#           <td>7.12</td>
#           <td>5.002</td>
#           <td>193.1</td>
#           <td>197.5</td>
#           <td>222.7</td>
#           <td>218.3</td>
#           <td>13.8</td>
#           <td>9.96</td>
#           <td>179.2</td>
#           <td>187.4</td>
#           <td>236.6</td>
#           <td>228.4</td>
#           <td>20.15</td>
#           <td>13.9</td>
#           <td>166.0</td>
#           <td>179.0</td>
#           <td>249.8</td>
#           <td>236.8</td>
#           </tr>
#           <tr>
#           <td><b>USL</b></td>
#           <td>0.79</td>
#           <td>1.07</td>
#           <td>639.26</td>
#           <td>337.46</td>
#           <td>649.44</td>
#           <td>651.24</td>
#           <td>9.303</td>
#           <td>6.67</td>
#           <td>585.95</td>
#           <td>602.95</td>
#           <td>706.15</td>
#           <td>689.15</td>
#           <td>17.95</td>
#           <td>12.46</td>
#           <td>530.05</td>
#           <td>565.85</td>
#           <td>762.05</td>
#           <td>726.25</td>
#           <td>22.6</td>
#           <td>17.34</td>
#           <td>500.05</td>
#           <td>534.05</td>
#           <td>792.05</td>
#           <td>758.05</td>
#           </tr>
#           <tr>
#           <td><b>APLLTD</b></td>
#           <td>1.8</td>
#           <td>2.07</td>
#           <td>956.62</td>
#           <td>953.99</td>
#           <td>991.68</td>
#           <td>994.31</td>
#           <td>8.26</td>
#           <td>6.27</td>
#           <td>922.95</td>
#           <td>942.95</td>
#           <td>1089.15</td>
#           <td>1069.15</td>
#           <td>16.4</td>
#           <td>11.13</td>
#           <td>841.05</td>
#           <td>894.05</td>
#           <td>1171.05</td>
#           <td>1118.05</td>
#           <td>24.65</td>
#           <td>17.39</td>
#           <td>758.05</td>
#           <td>831.05</td>
#           <td>1254.05</td>
#           <td>1181.05</td>
#           </tr>
#           <tr>
#           <td><b>IDEA</b></td>
#           <td>3.2</td>
#           <td>3.84</td>
#           <td>9.63</td>
#           <td>9.568</td>
#           <td>10.27</td>
#           <td>10.332</td>
#           <td>22.54</td>
#           <td>16.65</td>
#           <td>7.63</td>
#           <td>8.21</td>
#           <td>12.07</td>
#           <td>11.49</td>
#           <td>39.39</td>
#           <td>31.27</td>
#           <td>5.97</td>
#           <td>6.77</td>
#           <td>13.73</td>
#           <td>12.93</td>
#           <td>58.27</td>
#           <td>45.89</td>
#           <td>4.11</td>
#           <td>5.33</td>
#           <td>15.59</td>
#           <td>14.37</td>
#           </tr> 
#         </table>
#     </body>
# </html>
# </center>

# ## <center>  This marks the end of the notebook</center>

# <b>References:</b>
# [1]. https://financetrain.com/calculating-var-using-monte-carlo-simulation/
# [2]. https://risk-engineering.org/VaR/

# In[ ]:




