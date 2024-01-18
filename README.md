# Bayesian Machine Learning – Stochastic Volatility Model:

## Metadata:

This repo includes the Python code for a Stochastic Volatility Model of Apple stock returns from 2007-2022.

I developed this project for the DS6040 (Bayesian Machine Learning) Data Science masters program at the University of Virginia.

## Synopsis:

Apple stock returns exhibit high time-varying volatility – quite stable sometimes, highly variable at others, and the distribution of market returns is highly non-gaussian. Thus, sampling Apple stock volatilities would be very difficult.

Stochastic Volatility (SV) models help capture such dynamic volatility, creating a more realistic representation of changing market risks. As a Bayesian method, these models provide uncertainty estimates, enabling better risk management.

To develop the model, I pull the data from the `yfinance` package into a Pandas dataframe and calculate a 'returns' column as the difference in the logarithmic returns, which I then use to model the volatility.

I model the logarithm of the daily returns with a Student-T distribution, parameterized by:
- the degrees of freedom ($𝜈$) following an exponential distribution 
- volatility ($𝑠_𝑖$), where ($𝑖$) is the time index 

The volatility follows a Gaussian random walk across all time steps, parameterized by a common variance given by an exponential distribution.

I model the logarithmic returns at each time point. 

The model allows the volatility to change over time, such that the volatility at each time point is controlled by a parameter for that time point ($𝑠_𝑖$). 

However, the scale parameters ($𝑠_𝑖$) at each time point cannot be completely independent; otherwise, the model would overfit the data.

One thing worth noting is that I have a single variance (𝜎) for the volatility process across all time, which may not represent the true nature of stock return behavior.

![Alt text](https://github.com/WD-Scott/Stochastic-Volatility-Model/blob/main/model_platnotation.png)

I use the `PyMC` package to develop the SV model by writing a basic function that takes the Pandas dataframe as its input and returns the PyMC model. The model is parametrized by the stochastic process previously described to capture the volatility dynamics.

```python
def sv_model(data):
    with pm.Model(coords={"time": data.index.values}) as model:
        𝜎 = pm.Exponential("𝜎", 10)
        volatility = pm.GaussianRandomWalk(
            "volatility", sigma=𝜎, 
            dims="time", init_dist=pm.Normal.dist(0, 100)
        )
        𝜈 = pm.Exponential("𝜈", 0.1)
        r = pm.StudentT(
            "r", nu=𝜈, 
            lam=np.exp(-2 * volatility), 
            observed=data["return"], dims="time"
        )
    return model


svol_model = sv_model(df)
```

The notebook includes several visualizations with markdown cells above them providing descriptions.

## Data:

I pulled the Apple stock the Yahoo Finance package for 2007 through 2022.

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
