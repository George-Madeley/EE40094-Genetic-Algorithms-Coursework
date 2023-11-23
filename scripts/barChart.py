import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fileName = './results/test.csv'

df = pd.read_csv(fileName)


# We want to plot a bar chart of the num_generations of each selection_method and eltitism combination.
# We can do this by using the seaborn barplot function.

# We need to convert the elitism column to a string so that we can use it as a hue.
df['elitism'] = df['elitism'].astype(str)
sns.barplot(x='selection_method', y='num_generations', hue='elitism', data=df)
plt.title('Number of Generations vs Selection Method')
plt.xlabel('Selection Method')
plt.ylabel('Number of Generations')
plt.ylim(110, 140)
# add values to the top of each bar
for p in plt.gca().patches:
    plt.gca().annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
                       textcoords='offset points')
plt.show()


