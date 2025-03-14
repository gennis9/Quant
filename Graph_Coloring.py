'''
Initiated Date    : 2024/10/18
Last Updated Date : 2024/10/27
Aim: Minimize the total phases (waiting time) with no conflicting routes (safety).
Input Data: Network with nodes and links.
'''

# %% Enviornment

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx
import statsmodels.api as sm
from copy import deepcopy


## Set the path of data
_Path = r'D:\03Programs_Clouds\Google Drive\NSYSU\03Optimization'
os.chdir(_Path)

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)




# %% Data Access

## Read all edge files.
Data_FileNames = os.listdir(_Path)
## Only read txt file.
Data_FileNames = [file for file in Data_FileNames if 'pdf' not in file]

Raw_Edges = [pd.read_csv(Data_FileNames[j], sep='\s+', header=None, skiprows=2, names=['Node1', 'Node2']) for j in range(len(Data_FileNames))]



## Create the sheet to record the results.
idx = [ name[:-4] for name in Data_FileNames ]
Results = pd.DataFrame(np.nan, index = idx, columns = ['Initial 1', 'Initial 2', 'Initial 3', 'DSatur'])
Num_iter = pd.DataFrame(np.nan, index = idx, columns = ['Initial 1', 'Initial 2', 'Initial 3'])


# %% Centrality, Ordering, and Color Assigning

for network in range(len(Data_FileNames)):
    # network = 0

    ## Extract the nodes of current file.
    Edges = Raw_Edges[ network ]
    Nodes = pd.concat([Edges['Node1'], Edges['Node2']], ignore_index=True)
    
    ## Count the degree centrality.
    Centrality = Nodes.value_counts()
    
    ## Rank all nodes in a list.
    # Original sequence, Number of conflicts from low to high, Number of conflicts from high to low.
    Orders = [pd.DataFrame( Centrality.sort_index() ),
              pd.DataFrame( Centrality.sort_values() ),
              pd.DataFrame( Centrality.sort_values(ascending = False) )
             ]

    
    for method in range(3):
        order = Orders[ method ]
    
        ## Initialiation.
        order['Color'] = -999     # Uncolored
        color = 0
        
        ## Only handle the uncolored nodes.
        idx_uncolored = order[ order['Color'] < 0 ].index
        
        while len(idx_uncolored) > 0:
            ## Assign next color to be filled.
            color += 1
            
            ## Color nodes by a specific color as many as possible each time.
            for node in idx_uncolored:    
                ## Extract all edges that link to current nodes.
                nodes_adj = Edges[Edges.isin([ node ]).any(axis=1)]
                ## Extract the nodes whose color is current color.
                nodes_sameColor = order.index[ order['Color'] == color ]
                
                ## Check whether there is any neighbor linked colored with current color.
                if not nodes_sameColor.isin(nodes_adj.stack().values).any():
                    order.loc[node, 'Color'] = color
                    
            ## Only handle the uncolored nodes.
            idx_uncolored = order[ order['Color'] < 0 ].index
            
            print(len(idx_uncolored), '/', len(order))
            
        ## Record the number of colors used in current while loop.
        Results.iloc[network, method] = color



# %% Network Visualization

## Take the WP algorithm with 100 nodes as the example.
network = 3
method = 2

## Extract the nodes of current file.
Edges = Raw_Edges[ network ]
Nodes = pd.concat([Edges['Node1'], Edges['Node2']], ignore_index=True)

## Count the degree centrality.
Centrality = Nodes.value_counts()

## Rank all nodes in a list.
# Original sequence, Number of conflicts from low to high, Number of conflicts from high to low.
Orders = [pd.DataFrame( Centrality.sort_index() ),
          pd.DataFrame( Centrality.sort_values() ),
          pd.DataFrame( Centrality.sort_values(ascending = False) )
         ]


order = Orders[ method ]

## Initialiation.
order['Color'] = -999     # Uncolored
color = 0

## Only handle the uncolored nodes.
idx_uncolored = order[ order['Color'] < 0 ].index

while len(idx_uncolored) > 0:
    ## Assign next color to be filled.
    color += 1
    
    ## Only handle the uncolored nodes.
    idx_uncolored = order[ order['Color'] < 0 ].index
    ## Color nodes by a specific color as many as possible each time.
    for node in idx_uncolored:    
        ## Extract all edges that link to current nodes.
        nodes_adj = Edges[Edges.isin([ node ]).any(axis=1)]
        ## Extract the nodes whose color is current color.
        nodes_sameColor = order.index[ order['Color'] == color ]
        
        ## Check whether 
        if not nodes_sameColor.isin(nodes_adj.stack().values).any():
            order.loc[node, 'Color'] = color
            
    ## Only handle the uncolored nodes.
    idx_uncolored = order[ order['Color'] < 0 ].index
    

## Generate the network.
gr_100 = nx.Graph()
# gr_100.add_nodes_from(Nodes.unique())
gr_100.add_edges_from( zip(Edges['Node1'].tolist(), Edges['Node2'].tolist()) )




## Set parameters for the visualization.
plt.figure(figsize = (37, 16))
sizes = 10 + order['count']**2.5 * 20
colors = order['Color']
layout = nx.spring_layout(gr_100,seed=32)  #105 #32 #942
# layout = nx.drawing.nx_agraph.graphviz_layout(gr_100, prog='twopi')


nx.draw(gr_100, node_size=sizes, with_labels=False, node_color=colors, cmap='hsv', edgecolors='black', pos=layout)
plt.show()




# %% Analysis

## Regression w.r.t. color.
Reg = deepcopy(Results)

## Build the independent variables.
Reg['Edges'] = [ len(file) for file in Raw_Edges ]
Reg['Nodes'] = [ int(x[4:]) for x in Reg.index ]

Reg = pd.melt(Reg, id_vars=['Edges', 'Nodes'], value_vars=Results.columns, 
              var_name='Ordering', value_name='Color')

Reg['Density'] = 2 * Reg['Edges'] / Reg['Nodes'] / (Reg['Nodes'] - 1)


Reg = pd.get_dummies(Reg, columns=['Ordering'])
Reg.iloc[:, -3:] = Reg.iloc[:, -3:].astype(int)

## Use log-log regression.
Reg['Nodes'] = np.log(Reg['Nodes'])
Reg['Color'] = np.log(Reg['Color'])
Reg['Density'] = np.log(Reg['Density'])

## Set the random initialization as the benchmark.
model = sm.OLS( Reg['Color'], sm.add_constant(Reg.iloc[:, [1, 3, 5, 6]]) ).fit()
model.summary()


## Correlations of different ordering.
pd.Series(order.index).corr(order['count'])


## Use log-log regression to analysis overhead.
Overhead = pd.read_excel('Test.xlsx')
Overhead['Node'] = np.log(Overhead['Node'])
Overhead['Density'] = np.log(Overhead['Density'])
Overhead['Overhead'] = np.log(Overhead['Overhead'])

model = sm.OLS( Overhead['Overhead'], sm.add_constant(Overhead.iloc[:, [0, 2, 3, 4]]) ).fit()
model.summary()




# %% DSatur Algorithm

Corr = []

for network in range(len(Data_FileNames)):
    # network = 2
    
    ## Extract the nodes of current file.
    Edges = Raw_Edges[ network ]
    Nodes = pd.concat([Edges['Node1'], Edges['Node2']], ignore_index=True)
    
    ## Count the degree centrality and rank all nodes.
    order = pd.DataFrame( Nodes.value_counts().sort_values(ascending = False) )
    
    ## Initialiation.
    order['Color'] = -999     # Uncolored
    color = 0
    # Represent the degree of color saturation for a node.
    order['Saturation'] = 0
    
    ## Color the first subgraph of the node with the highest centrality by WP algorithm.
    node = order.index[0]
    neighbor = Edges[Edges.isin([ node ]).any(axis=1)]
    neighbor = order.loc[ list(set(neighbor.stack().values)) ]
    neighbor = neighbor.sort_values(by = ['count'], ascending = False)
    idx_uncolored = neighbor.index
    
    while len(idx_uncolored) > 0:
        ## Assign next color to be filled.
        color += 1
        
        ## Color nodes by a specific color as many as possible each time.
        for node in idx_uncolored:    
            ## Extract all edges that link to current node.
            nodes_adj = Edges[Edges.isin([ node ]).any(axis=1)]
            ## Extract the nodes whose color is current color.
            nodes_sameColor = order.index[ order['Color'] == color ]
            
            ## Check whether there is any neighbor linked colored with current color.
            if not nodes_sameColor.isin(nodes_adj.stack().values).any():
                order.loc[node, 'Color'] = color
                neighbor.loc[node, 'Color'] = color
    
        ## Only handle the uncolored nodes.
        idx_uncolored = neighbor[ neighbor['Color'] < 0 ].index
        
        print(len(idx_uncolored), '//', len(neighbor))
    
    
    ## For remaining nodes.
    idx_uncolored = order[ order['Color'] < 0 ].index
    for loop in range( len(idx_uncolored) ):
        ## Update the degree of color saturation for the uncolored nodes.
        for node in idx_uncolored:
            ## Extract all edges that link to current node.
            nodes_adj = Edges[Edges.isin([ node ]).any(axis=1)].stack().values
            ## Count how many different colors is linked to current node.
            order.loc[node, 'Saturation'] = len( order.loc[ list(set(nodes_adj)), 'Color' ][ order['Color'] > 0 ].unique() )
            
        ## Sort by Saturation (descending) and Degree (descending) to find the best node.
        uncolored_nodes = order.loc[idx_uncolored].sort_values(by=['Saturation', 'count'], ascending=[False, False])
        node = uncolored_nodes.index[0]
    
    
        ## Extract all edges that link to current node.
        nodes_adj = Edges[Edges.isin([ node ]).any(axis=1)].stack().values
        ## Extract the colors that link to current node.
        colors_adj = order.loc[ list(set(nodes_adj)), 'Color' ][ order['Color'] > 0 ].unique()
        
        color = 1
        ## Force to assign a color to current node without contradiction to the adjacent nodes.
        while order.loc[node, 'Color'] < 0:
            ## If current color exists in any of the adjacent nodes, assign next color.
            if color in colors_adj:
                color += 1
            else:
                order.loc[node, 'Color'] = color
                
        ## Only handle the uncolored nodes.
        idx_uncolored = order[ order['Color'] < 0 ].index
        
        print(len(idx_uncolored), '/', len(order))
    
    
    ## Record the number of colors used in current while loop.
    Results.iloc[network, 3] = order['Color'].max()
    
    
    ## Update the degree of color saturation and to see the correlation between the degree centrality.
    for node in order.index:
        ## Extract all edges that link to current node.
        nodes_adj = Edges[Edges.isin([ node ]).any(axis=1)].stack().values
        ## Count how many different colors is linked to current node.
        order.loc[node, 'Saturation'] = len( order.loc[ list(set(nodes_adj)), 'Color' ][ order['Color'] > 0 ].unique() )
    
    
    Corr.append( order['count'].corr(order['Saturation']) )

