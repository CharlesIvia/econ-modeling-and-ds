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

# Exponential Moving Average 

- The exponential moving average is a widely used method to filter out noise and identify trends. The weight of each element decreases progressively over time, meaning the exponential moving average gives greater weight to recent data points. This is done under the idea that recent data is more relevant than old data.

- Compared to the simple moving average, the exponential moving average reacts faster to changes, since is more sensitive to recent movements.

- EMAt = { axt + (1- α)EMAt- 1 :  t =0, t > 0

-Where: 

- xₜ is the observation at the time period t.
- EMAₜ is the exponential moving average at the time period t.
- α is the smoothing factor. The smoothing factor has a value between 0 and 1 and represents the weighting applied to the most recent period.

