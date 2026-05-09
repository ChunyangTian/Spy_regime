#pip install yfinance pandas matplotlib numpy hmmlearn

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn.hmm import GaussianHMM


#=== Part 1 ===
#Download the data from yahoo finance for S&P500
data = yf.download('SPY', start='2004-01-01', end='2026-05-01')
data.columns = data.columns.get_level_values(0)
df = data.loc[:, ['Close']]


# calculation:
# 1.Daily log returns
# 2.Annualized volatility(21 & 252 avoid weekend)
df['Return'] = np.log(df['Close'] / df["Close"].shift(1))
df['Volatility'] = df['Return'].rolling(window=21).std() * np.sqrt(252)
df.dropna(inplace = True)

print(df)

#transform the data into a matrix for HMM
X = df[['Return', 'Volatility']].values


#=== Part 2 ===
# Choose the number of regimes by BIC
# Lower BIC is better; it punishes models with too many parameters
print("\n==== Choosing number of regimes =====")
for n in range(2, 6):
    try:
        test_model = GaussianHMM(n_components=n, covariance_type='full',
                                 n_iter=1000, random_state=42)
        test_model.fit(X)
        log_likelihood = test_model.score(X)

        n_features = X.shape[1]
        n_parameters  = n - 1
        n_parameters += n * (n - 1)
        n_parameters += n * n_features
        n_parameters += n * n_features * (n_features + 1) / 2
        bic = -2 * log_likelihood + n_parameters * np.log(len(X))
        print(f"Number of regimes: {n}, Log Likelihood: {log_likelihood:.2f}, BIC: {bic:.2f}")
    except Exception as e:
        print(f"Number of regimes: {n}, model failed to fit ({type(e).__name__})")

model = GaussianHMM(n_components = 4, covariance_type = 'full', n_iter = 1000,random_state=42)
model.fit(X)
regimes = model.predict(X)
df['Regime'] = regimes

# === Part 3 ===
# Validate the model: does the crisis regime capture known market crises?
# Step 1: Find which regime is the crisis regime (highest return volatility)
print("\n==== Crisis capture check ====")

regime_0_vol = df[df['Regime'] == 0]['Return'].std()
regime_1_vol = df[df['Regime'] == 1]['Return'].std()
regime_2_vol = df[df['Regime'] == 2]['Return'].std()
regime_3_vol = df[df['Regime'] == 3]['Return'].std()  

print(f"Regime 0 vol: {regime_0_vol:.4f}")
print(f"Regime 1 vol: {regime_1_vol:.4f}")
print(f"Regime 2 vol: {regime_2_vol:.4f}")
print(f"Regime 3 vol: {regime_3_vol:.4f}")         
if regime_0_vol >= regime_1_vol and regime_0_vol >= regime_2_vol and regime_0_vol >= regime_3_vol:
    crisis_regime = 0
elif regime_1_vol >= regime_2_vol and regime_1_vol >= regime_3_vol:
    crisis_regime = 1
elif regime_2_vol >= regime_3_vol:
    crisis_regime = 2
else:
    crisis_regime = 3

print(f"\nCrisis regime is Regime {crisis_regime}\n")

# Step 2: Check each known crisis window
# 2008 Global Financial Crisis
window = df.loc['2008-09-01':'2009-03-31']
in_crisis = (window['Regime'] == crisis_regime).sum()
total = len(window)
print(f"2008 GFC:    {in_crisis}/{total} days in crisis regime ({in_crisis/total*100:.0f}%)")

# 2020 COVID crash
window = df.loc['2020-02-20':'2020-04-30']
in_crisis = (window['Regime'] == crisis_regime).sum()
total = len(window)
print(f"2020 COVID:  {in_crisis}/{total} days in crisis regime ({in_crisis/total*100:.0f}%)")

# 2022 Bear market
window = df.loc['2022-04-01':'2022-10-31']
in_crisis = (window['Regime'] == crisis_regime).sum()
total = len(window)
print(f"2022 Bear:   {in_crisis}/{total} days in crisis regime ({in_crisis/total*100:.0f}%)")

# 2025 Tariff selloff
window = df.loc['2025-04-01':'2025-06-30']
in_crisis = (window['Regime'] == crisis_regime).sum()
total = len(window)
print(f"2025 Tariff: {in_crisis}/{total} days in crisis regime ({in_crisis/total*100:.0f}%)")

# 2026 Iran War
window = df.loc['2026-02-28':'2026-05-07']
in_crisis = (window['Regime'] == crisis_regime).sum()
total = len(window)
print(f"2026 Iran War: {in_crisis}/{total} days in crisis regime ({in_crisis/total*100:.0f}%)")

# === Part 4 ===
# Compute summary statistics for each regime (mean and std of return and volatility)
# After running, regimes can be interpreted by their annualized return and volatility:
# - Slow Bull: high return, low vol
# - Crisis: very negative return, very high vol  
# - Normal Bull: moderate return, moderate vol
# - Volatile Correction: small positive return, elevated vol
df_summary = df.groupby('Regime').agg({
    'Return': ['mean', 'std'],
    'Volatility': ['mean', 'std'],
})
df_summary.columns = ['daily_return_mean', 'daily_return_std', 'volatility_mean', 'volatility_std']
df_summary['annualized_return_mean'] = df_summary['daily_return_mean'] * 252
df_summary['annualized_return_std'] = df_summary['daily_return_std'] * np.sqrt(252)

plot_table = df_summary.round(4).reset_index()

fig,(ax1,ax2) = plt.subplots(2,1, figsize=(16,10), gridspec_kw={'height_ratios':[3,1]})

for i in range(model.n_components):
    mask = df['Regime'] == i
    ax1.scatter(df.index[mask], df['Close'][mask], label=f'Regime {i}', s=10)

ax1.set_title('S&P500 Price with Regimes')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, alpha=0.2)

ax2.axis('off')
table = ax2.table(cellText=plot_table.values, colLabels=plot_table.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.tight_layout()
plt.show()