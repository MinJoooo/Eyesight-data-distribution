import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import math


df = pd.read_csv('data.csv')

mean, std = stats.norm.fit(df['left'])
print('[left]')
print('mean:', round(mean, 3))
print('std:', round(std, 3), '\n')

mean, std = stats.norm.fit(df['right'])
print('[right]')
print('mean:', round(mean, 3))
print('std:', round(std, 3))

fig, axes = plt.subplots(ncols=len(df.columns), figsize=(10,5))
for col, ax in zip(df, axes):
    df[col].value_counts().sort_index().plot.bar(ax=ax, title=col)

plt.tight_layout()
plt.show()


# normal distribution
mu = mean
variance = std*std
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()


# log normal distribution
mu = mean
sigma = std
s = np.random.lognormal(mu, sigma, 1000)

count, bins, ignored = plt.hist(s, 100, density=True, align='mid')
x = np.linspace(min(bins), max(bins), 10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
       / (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show()