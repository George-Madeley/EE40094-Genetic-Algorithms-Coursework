import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

file_name = './results/retain_with_population.csv'
column_name = 'retain'
y_axis_label = 'Retain'

df = pd.read_csv(file_name)

minVal = math.ceil(df["num_generations"].min())
maxVal = math.floor(df["num_generations"].max())
upperQuartile = df["num_generations"].quantile(0.75)
center = df["num_generations"].mean()
median = df["num_generations"].median()

print("Min: " + str(minVal))
print("Max: " + str(maxVal))
print("Mean: " + str(center))
print("Median: " + str(median))

colors = sns.diverging_palette(0, 120, s=100, l=50, n=9, as_cmap=True)
colors = sns.blend_palette(["green", "yellow", "red"], as_cmap=True)
results = df.pivot(index="population_count", columns=column_name, values="num_generations")

sns.heatmap(
  results,
  vmin=minVal,
  vmax=upperQuartile,
  cmap=colors,
  square=True,
)
plt.xlabel(y_axis_label)
plt.ylabel('Population Count')
plt.title(f'{y_axis_label} vs Population Count')
plt.show()