import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_name = './results/retain.csv'
x_axis_label = 'Retain'
column_name = 'retain'

# Read in the data
df = pd.read_csv(file_name)

# Plot the data using a log scale on x axis
plt.plot(df[column_name], df['num_generations'])
# plt.xscale('log')
plt.xlabel(x_axis_label)
plt.ylabel('Number of Generations')
plt.title(f'{x_axis_label} vs Number of Generations')
plt.show()
