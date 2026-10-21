# Detecting S&P 500 Market Regimes Using Hidden Markov Model

**Author:** Chunyang Tian 
**Date:** May 2026  
**Goal:** Identify hidden market regimes (calm bull, normal bull, volatile, crisis) 
in S&P 500 daily data using a 4-state Gaussian HMM, and validate against known 
historical crises.

## Purpose & Outcome

This model aim to find current US stock market status. S&P 500 (SPY) is used as 
the reference for this model, but other market ETFs also work with this model.

After test with BIC (Bayesian Information Criterion), we found 4 regime is the 
best balance for our data. They are called "Regime 0/1/2/3" in the code, and their 
characteristic could identify by annualized return and volatility in the table 
under the chart.

The 4 regimes can be interpret as:
- **Slow Bull**: Highest annualized return (~17%) and lowest volatility (~9%). 
  Long stable uptrends like 2017 or much of 2024.
- **Crisis**: Lowest annualized return (~-43%) and highest volatility (~54%). 
  Very rare, only ~5% of all trading days.
- **Normal Bull**: Moderate return (~11%) with moderate volatility (~14%). 
  The most common state.
- **Volatile Correction**: Small positive return (~8%) with elevated 
  volatility (~22%). Markets is shaky but not crashing.

## How the model work

1. Download SPY daily data from Yahoo Finance (2004 to 2026).
2. Calculate two features for each day:
   - Daily log return: `log(Close_today / Close_yesterday)`
   - Annualized realized volatility: rolling 21-day standard deviation * sqrt(252)
3. Use these two features to fit a Gaussian Hidden Markov Model.
4. The HMM assume the market is in one of several "hidden states" each day, and 
   each state produce return and volatility from its own gaussian distribution. 
   The model find these states automatically without us telling it.

## Why I choose 4 regimes (BIC test)

I tested k = 2, 3, 4, 5 by minimizing BIC (lower is better)
k = 5 actually crashed because one regime's covariance matrix become non-positive-definite. 
This may mean the data is not enough to support 5 separate states, or the fitting just 
got stuck. So among the models that fit, 4 regimes has the lowest BIC, and I stop at 4.

**About BIC = -2 * log_likelihood + n_parameters * np.log(len(X)):**

Left side of plus sign is the reward term. Likelihood means "how likely this 
data is to happen under our model". Bigger likelihood = better fit. The -2 is 
just to flip direction so smaller value means better fit, which match the rule 
"smaller BIC is better". Log is to keep the number stable, otherwise likelihood 
is product of many small probabilities and will underflow to 0.

Right side is the punish term. Bigger model (more parameters) gets bigger punish, 
to prevent overfitting. n_parameter counts the parameters in our HMM:

1. **Initial probability** (which regime on day 1): we need n-1 because once you 
know n-1 of them, the last one is automatic since sum of probability is 1.

2. **Transition matrix** (regime today to tomorrow): take n=3 as example, today 
has 3 possibility, tomorrow also 3, so the matrix is 3*3 = 9. But each row sum 
to 1, same freedom concept, so each row only n-1 = 2 free. Total: n * (n-1).

3. **Mean of each regime**: each regime's distribution has a 2D mean point 
(mean_Return, mean_Volatility). Total: n * n_features.

4. **Covariance of each regime**: each regime has a 2x2 covariance matrix. The 
matrix is symmetric so cov(R,V) = cov(V,R), only count once. Independent 
elements = n_features * (n_features+1) / 2. Total: n * n_features * (n_features+1) / 2.

## Test on past market crises

I checked how many days in known crisis windows are labeled as the crisis regime:

| Crisis | Days in crisis regime | Capture rate |
|---|---|---|
| 2008 GFC (Sep 2008 - Mar 2009) | 137/146 | 94% |
| 2020 COVID (Feb 20 - Apr 30) | 43/50 | 86% |
| 2022 Bear Market (Apr - Oct) | 21/147 | 14% |
| 2025 Tariff Selloff (Apr - Jun) | 25/62 | 40% |
| 2026 Iran War (Feb 28 - May 7) | 0/43 | 0% |

This result is interesting and not what I expect at first. The model capture sharp 
crashes very well (2008, 2020) but miss slow bear market (2022) and geopolitical 
events (2026 Iran War). 

The 2026 Iran War result is most surprising. Even though the war is a serious 
event with oil supply disruption, S&P 500 actually reached new record high during 
this window (over 7000), and the volatility stay around 11-18%, which is far from 
crisis level (~50% vol). 

This show that the model is detecting **market panic**, not **news severity**. If 
the market does not panic (like 2026 Iran War, where investors think it will be 
short), the model won't classify it as crisis, even if the news is very serious.

## Limitations

Some limits I know about:

1. **This is descriptive, not predictive.** The model is fit on the entire 2004-2026 
   data, so when it says "this day is crisis regime", it already saw the future when 
   making this label. A real forecasting test would need rolling refit (train on data 
   up to time t, predict t+1, then move forward), which I did not do here.

2. **Volatility is a lagged feature.** Because volatility is computed from past 21 days, 
   the crisis regime usually appears at the same time as the crash, not before it. 
   So even if I do rolling refit, the "prediction" would be late.

3. **Only 2 features is not enough.** The Iran War result shows the model misses 
   events that don't show up in price/volatility. To detect events the market is 
   ignoring (or events before the market reacts), we may need other features like 
   VIX (option implied vol), credit spreads, or news-based indicators.

4. **The label of regimes (0/1/2/3) is random.** HMM doesn't know which regime is 
   "crisis", so I have to find it manually by looking at which regime has highest 
   return std. Every time the random_state change, the labels may switch.

## Future work

- Try rolling out-of-sample fitting to test real predictive power.
- Add more features (VIX, momentum indicators, yield curve) to help detect non-panic 
  type stress.

## How to run

```bash
pip install yfinance pandas matplotlib numpy hmmlearn
python S&P_regime.py
```

The script will print BIC table, regime statistics, crisis capture rates, and show 
a plot with price colored by regime + a summary table.
