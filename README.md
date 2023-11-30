# Bayesian Machine Learing -- Stochastic Volatility Model:

## Metadata:

This repo includes the Python code for a Stochastic Volatility Model of Apple stock returns from 2007-2022.

I developed this project for the DS6040 (Bayesian Machine Learning) Data Science masters program at the University of Virginia.

## Synopsis:

Apple stock returns exhibit high time-varying volatility – quite stable sometimes, highly variable at others, and the distribution of market returns is highly non-gaussian. Thus, sampling Apple volatilities would be very difficult.

Stochastic Volatility (SV) models help capture such dynamic volatility, creating a more realistic representation of changing market risks. As a Bayesian method, these models provide uncertainty estimates, enabling better risk management.

To develop the model, I pull the data from the `yfinance` package into a Pandas dataframe and calculate a 'returns' column as the difference in the logarithmic returns, which I then use to model the volatility.

I model the logarithm of the daily returns with a Student-T distribution, parameterized by:
- the degrees of freedom ($𝜈$) following an exponential distribution 
- volatility ($𝑠_𝑖$), where ($𝑖$) is the time index 

The volatility follows a Gaussian random walk across all time steps, parameterized by a common variance given by an exponential distribution. 

I model the logarithmic returns at each timepoint. 

The model allows the volatility to change over time, such that the volatility at each time point is controlled by a parameter for that time point ($𝑠_𝑖$). 

But, the scale parameters ($𝑠_𝑖$) at each timepoint cannot be completely independent, otherwise, the model would overfit the data.

One thing worth noting is that I have a single variance (𝜎) for the volatility process across all time, which may not be representative of the true nature of stock return behavior.

I use the `PyMC` package to develop the SV model by writing a basic function that takes the Pandas dataframe as its input and returns the PyMC model. The model is parametrized by the stochastic process previously described to capture the volatility dynamics.

The notebook includes several visualizations with markdown cells above them providing descriptions.


## Data:

Apple stock data were pulled from the Yahoo Finance (`Yfinance`) Python library for dates roughly ranging from 2007 to 2022.

- - - -
## Main Packages and Dependencies:

`PyMC`:     5.6.1

`Numpy`:    1.23.5

`Pandas`:   2.0.3

`Arviz`:    0.16.1

`Sklearn`:  1.3.0
- - - -

## Manifest:

### Files in repo:
* SVM.ipynb
* apple2.csv
* LICENSE
* README.md
