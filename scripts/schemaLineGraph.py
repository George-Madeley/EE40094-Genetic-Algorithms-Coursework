import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_name = './results/schema_theorem.csv'
x_axis_label = 'Num Generations'

# Read in the data
df = pd.read_csv(file_name)

# Plot the data using a log scale on x axis
plt.plot(df['num_generations'], df['num_matches'])
plt.plot(df['num_generations'], df['num_expected_matches'])
# plt.xscale('log')
plt.xlabel('Number of Generations')
plt.ylabel('Number of Matches')
plt.title(f'Number of Schema Matches vs Number of Generations')
plt.legend(['Actual', 'Expected'])
plt.show()
