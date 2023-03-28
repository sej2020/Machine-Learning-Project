# Import libraries
import matplotlib.pyplot as plt
import numpy as np
 
 
# Creating dataset
np.random.seed(10)
 
data_1 = np.random.normal(4, .15, 30)
data_2 = np.random.normal(3.75, .08, 30)
data_3 = np.random.normal(3.8, .25, 30)
data_4 = np.random.normal(4.1, .51, 30)
data_5 = np.random.normal(4.05, .12, 30)
data = [data_1, data_2, data_3, data_4, data_5]

 
styledict= {'boxprops': {'linestyle': '-', 'linewidth': 1, 'color': 'black'},
            'flierprops': {'marker': 'D', 'markerfacecolor': 'white', 'markersize': 4, 'linestyle': 'none'},
            'medianprops': {'linestyle': '-.', 'linewidth': 1, 'color': 'black'},
            'whiskerprops': {'linestyle': '--', 'linewidth': 1, 'color': 'black'},
            'capprops': {'linewidth': 1, 'color': 'black'}, 'boxfill': 'lightgray', 'grid': True, 'dpi': 300.0}

# Creating plot
boxfig = plt.figure()
ax = boxfig.add_subplot(111)
bp = ax.boxplot(data, patch_artist = True, vert = 0, boxprops = styledict['boxprops'],
                flierprops = styledict['flierprops'], medianprops = styledict['medianprops'],
                whiskerprops = styledict['whiskerprops'], capprops = styledict['capprops']
                )

for patch in bp['boxes']:
    patch.set_facecolor(styledict['boxfill'])

ax.set_yticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])
ax.yaxis.grid(styledict['grid'])
ax.xaxis.grid(styledict['grid'])

plt.title('Predictability of Subsets of Training Data')

ax.set_xlabel(f'RMSE for Regressor Predictions')
ax.set_ylabel('Models')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# show plot
plt.show()



