# Simple, cumulative and exponential moving averages

# moving average  = rolling mean
 
- Useful in determining trends in finance

- Commonly used with time series data to smooth random short-term variations and highlight other components such as trend, season or cycle in the data

- Simple MA (SMA), Cumulative MA( CMA), Exponential MA (EMA)

# WANT: Explore SMA, CMA, EMA usinf rainfall and temperature data from Open Data Barcelona

# First - data pre-processing

- Dataset 1 = Monthly average air temp of Barcelona city since 1980
- Dataset 1 = Monthly accumulated air rainfall of Barcelona city since 1976

# The Simple Moving Average

- The simple moving average is the unweighted mean of the previous M data  points.

- The selection of M (sliding window) depends on the amount of smoothing desired since increasing the value of M improves the smoothing at the expense of accuracy.

- SMAt = Xt + Xt-1 + Xt-2 + ... Xm - (t-1) / M

- The size of the window (number of periods) is specified in the argument window.

- Rolling method can be used together with other statistical functions such as .rolling().min(), .rolling().max(),.rolling().median(),.rolling().std(),.rolling().var(),.rolling().sum(),

# Cumulative Moving Average 

- The Cumulative Moving Average is the unweighted mean of the previous values up to the current time t.

-CMAt = x1 + x2 + x3 + ... + xt /t

-The simple moving average has a sliding window of constant size M. On the contrary, the window size becomes larger as the time passes when computing the cumulative moving average

- From the results, the CMA is a bad option in anlayzing trends

