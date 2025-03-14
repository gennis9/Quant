'''
Initiated Date    : 2024/11/29
Last Updated Date : 2024/12/01
Aim: Solve traveling salesman problem by metaheuristics.
Input Data: Traveling salesman problem with 2-dimension coordinates.
'''

# %% Environment

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from copy import deepcopy
%matplotlib inline


## Set the path of data
# _Path = r'D:\03Programs_Clouds\Google Drive\NSYSU\03Optimization'
os.chdir(_Path)

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)




# %% Data Access

## Read all data files.
Data_FileNames = os.listdir(_Path)

Coordinates, Coord_Types = [], []
for _name in range(len(Data_FileNames)):
    with open(Data_FileNames[_name], 'r') as file:
        lines = file.readlines()
    start_index = next(i for i, line in enumerate(lines) if line.strip() == "NODE_COORD_SECTION") + 1
    end_index = next(i for i, line in enumerate(lines) if line.strip() == "EOF")
    
    file = pd.read_csv(Data_FileNames[_name], sep='\s+', header=None, 
                       skiprows=start_index, nrows=end_index - start_index)
    file.columns = ['index', 'x-coord', 'y-coord']
    Coordinates.append( file.iloc[:, 1:] )
    
    edge_weight_type = lines[4].strip().split(":")[1].strip()  # Get the value after "EDGE_WEIGHT_TYPE:"
    Coord_Types.append(edge_weight_type)




# %% Euclidean Distance Measurement

## Calculate Euclidean distance between two cities.
def Func_Euclidean_Distance(x1, x2, y1, y2, dist_type = 'EUC_2D'):
    if dist_type == 'GEO':
        ## Convert degrees to radians.
        lat1, lon1 = np.radians(y1), np.radians(x1)
        lat2, lon2 = np.radians(y2), np.radians(x2)
    
        ## Haversine formula.
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
        ## Compute the distance.
        # Earth's radius is 6371 km.
        return 6371 * c
    
    else:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

## Create a distance matrix for all cities.
def Func_Distance_Matrix(nodes, dist_type = 'EUC_2D'):
    n = len(nodes)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = Func_Euclidean_Distance(
                nodes.iloc[i, 0], nodes.iloc[j, 0],
                nodes.iloc[i, 1], nodes.iloc[j, 1],
                dist_type)
    return dist_matrix + dist_matrix.T

distance_matrices = [ Func_Distance_Matrix(file, dist_type) for file, dist_type in zip(Coordinates, Coord_Types) ]




# %% Initial Solution - Nearest Neighbor Algorithm

## Build the nearest/farthest neighbor algorithm.
def Func_Initial_Neighbor(dist_matrix, start_node=0, metrics='min'):
    n = len(dist_matrix)
    cities_unvisited = [ *range(n) ]
    cities_unvisited.remove( start_node )
    tour = [start_node]
    current_node = start_node
    total_dist = 0

    for _ in range(n - 1):
        temp_dist = []
        ## Find the nearest/farthest unvisited neighbor.
        for unvisited_node in cities_unvisited:
            temp_dist.append( dist_matrix[current_node][unvisited_node] )
        
        if metrics == 'min':
            ## Record the next nearest city to be visited.
            next_dist = min(temp_dist)
        else:
            ## Record the next farthest city to be visited.
            next_dist = max(temp_dist)
        
        next_node = cities_unvisited[ temp_dist.index( next_dist ) ]
        ## Collect the distance visited.
        total_dist += next_dist
        tour.append(next_node)
        current_node = next_node
        ## Remove the next city to be visited and remove it from unvisited list.
        cities_unvisited.remove( next_node )

    ## Return to start node to complete the tour.
    tour.append(start_node)
    total_dist += dist_matrix[current_node][start_node]
    return tour, total_dist


## Solve TSP using nearest and farthest neighbor algorithms to get initial solutions.
start_node = 0  # Assuming node 1 is the starting point
Tour_NN, Distance_NN, Tour_FN, Distance_FN  = [], [], [], []
for file in range(len(Coordinates)):
    ## Record the tour from nearest neighbor algorithm.
    tour = Func_Initial_Neighbor(distance_matrices[file], start_node=start_node, metrics='min')
    Tour_NN.append( tour[0] )
    Distance_NN.append( tour[1] )
    
    ## Record the tour from farthest neighbor algorithm.
    tour = Func_Initial_Neighbor(distance_matrices[file], start_node=start_node, metrics='max')
    Tour_FN.append( tour[0] )
    Distance_FN.append( tour[1] )




# %% Visualization of the Tour Paths

# Visualize the tour
def Func_Plot_Tour(cities, tour):
    plt.figure(figsize=(10, 6))
    for i in range(len(tour) - 1):
        start = tour[i]
        end = tour[i + 1]
        plt.plot([cities.iloc[start, 0], cities.iloc[end, 0]],
                 [cities.iloc[start, 1], cities.iloc[end, 1]], 'bo-')
    
    plt.scatter(cities['x-coord'], cities['y-coord'], c='red', s=50)
    ## Highlight the starting node
    plt.scatter(cities.iloc[start_node, 0], cities.iloc[start_node, 1], 
                c='red', s=200, label='Starting Node', edgecolors='black')
    
    ## Mark the cities near to the nodes.
    coord_norm_x = ( cities['x-coord'].max() - cities['x-coord'].min() ) * 0.01
    coord_norm_y = ( cities['y-coord'].max() - cities['y-coord'].min() ) * 0.01
    for i, row in cities.iterrows():
        plt.text(row['x-coord']-coord_norm_x, row['y-coord']+coord_norm_y, 
                 str(i+1), fontsize=8, ha='right')
    plt.title("Nearest Neighbor Tour")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

Func_Plot_Tour(Coordinates[-1], Tour_NN[-1])
Func_Plot_Tour(Coordinates[0], Tour_NN[0])




# %% 2-Opt Algorithm

def Func_Distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))

def Func_Two_Opt(tour, dist_matrix):
    n = len(tour)
    history = [ Func_Distance(tour, dist_matrix) ]
    
    improvement = True
    while improvement:
        improvement = False
        for i in range(1, n - 2):  # Avoid the starting node
            for j in range(i + 1, n - 1):  # Ensure a valid swap
                # Calculate the distances of the current and proposed edges
                d1 = dist_matrix[tour[i - 1]][tour[i]] + dist_matrix[tour[j]][tour[j + 1]]
                d2 = dist_matrix[tour[i - 1]][tour[j]] + dist_matrix[tour[i]][tour[j + 1]]
                # If swapping reduces the distance, apply the change
                if d2 < d1:
                    tour[i:j+1] = tour[i:j+1][::-1]  # Reverse the segment
                    improvement = True
                    history.append( Func_Distance(tour, dist_matrix) )
                    break  # Restart after the improvement
            if improvement:
                break
    return tour, history



Tour_NN_2Opt, Distance_NN_2Opt, Tour_FN_2Opt, Distance_FN_2Opt  = [], [], [], []
for file in range(len(Coordinates)):
    ## Record the tour from nearest neighbor algorithm.
    tour = Func_Two_Opt(Tour_NN[ file ], distance_matrices[ file ])
    Tour_NN_2Opt.append( tour[0] )
    Distance_NN_2Opt.append( tour[1] )
    
    ## Record the tour from farthest neighbor algorithm.
    tour = Func_Two_Opt(Tour_FN[ file ], distance_matrices[ file ])
    Tour_FN_2Opt.append( tour[0] )
    Distance_FN_2Opt.append( tour[1] )


Two_opt_NN, Two_opt_FN, Improvement_NN, Improvement_FN = [], [], [], []
for file in range(len(Coordinates)):
    temp = Distance_NN_2Opt[file][-1]
    Two_opt_NN.append( temp )
    diff = Distance_NN[file] - temp
    Improvement_NN.append( diff )
    
    temp = Distance_FN_2Opt[file][-1]
    Two_opt_FN.append( temp )
    diff = Distance_FN[file] - temp
    Improvement_FN.append( diff )



Distance_improved = pd.DataFrame({
    'NN': Distance_NN,
    'FN': Distance_FN,
    '2_Opt_NN': Two_opt_NN,
    '2_Opt_FN': Two_opt_FN,
    'Improvement_NN': Improvement_NN,
    'Improvement_FN': Improvement_FN,
    }, index = [ name[:-4] for name in Data_FileNames ]
    )



# %% Visualization of 2-Opt Convergence History

file = 8

plt.figure(figsize=(8, 5))
plt.plot(range(len(Distance_NN_2Opt[file])), Distance_NN_2Opt[file], marker='x', linestyle='-', color='b', label='NN')
plt.plot(range(len(Distance_FN_2Opt[file])), Distance_FN_2Opt[file], marker='x', linestyle='-', color='r', label='FN')
plt.title("Convergence History of " + Data_FileNames[file][:-4])
plt.xlabel("Number of Times of Convergence")
plt.ylabel("Total Distance of the Tour")
plt.grid(True)
plt.legend()
plt.show()


draw_NN, draw_FN = [], []
for i in range(1, len(Distance_NN_2Opt[file])):
    draw_NN.append( 100*(Distance_NN_2Opt[file][i-1] - Distance_NN_2Opt[file][i]) / Distance_NN_2Opt[file][i-1] )
    draw_FN.append( 100*(Distance_FN_2Opt[file][i-1] - Distance_FN_2Opt[file][i]) / Distance_FN_2Opt[file][i-1] )

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(Distance_NN_2Opt[file])), draw_NN, marker='x', linestyle='-', color='b', label='NN', alpha = 0.7)
plt.plot(range(1, len(Distance_NN_2Opt[file])), draw_FN, marker='x', linestyle='-', color='r', label='FN', alpha = 0.5)
plt.title("Convergence History in Percnetage of " + Data_FileNames[file][:-4])
plt.xlabel("Number of Times of Convergence")
plt.ylabel("Improvement (basis points)")
plt.grid(True)
plt.legend()
plt.show()



Data_FileNames[file][:-4]



# %% 3-Opt Algorithm

# 3-Opt Algorithm
def Func_Three_Opt(tour, dist_matrix):
    n = len(tour)
    history = [ Func_Distance(tour, dist_matrix) ]
    improvement = True

    while improvement:
        improvement = False
        for i in range(1, n - 3):  # Avoid the starting/ending node
            for j in range(i + 1, n - 2):
                for k in range(j + 1, n - 1):
                    ## Divide the tour by 3-opt.
                    segments = [tour[:i], tour[i:j], tour[j:k], tour[k:]]
                    ## Reverse the segments in combination under 3-opt.
                    # 2-opt is ruled out.
                    Options_3 = [
                        segments[0] + segments[1][::-1] + segments[2] + segments[3],
                        segments[0] + segments[1] + segments[2][::-1] + segments[3],
                        segments[0] + segments[1][::-1] + segments[2][::-1] + segments[3],
                        segments[0] + segments[2] + segments[1] + segments[3],
                        segments[0] + segments[2][::-1] + segments[1] + segments[3],
                        segments[0] + segments[2] + segments[1][::-1] + segments[3]
                        ]
                    
                    ## Compute the distance of the changing points among segments.
                    Distances_opt = [ Func_Distance(option, dist_matrix) for option in Options_3 ]
                    ## Find the segment under 3-opt generate shortest distance.
                    # When 3-opt does improve the distance shortening, update the tour.
                    if min(Distances_opt) < Func_Distance(tour, dist_matrix):
                        tour = Options_3[ Distances_opt.index(min(Distances_opt)) ]
                        improvement = True
                        history.append( Func_Distance(tour, dist_matrix) )
                        print('Update times:', len(history)-1)
                        break  # Restart after the improvement
                if improvement:
                    break
            if improvement:
                break
            
    return tour, history


Tour_NN_3Opt, Distance_NN_3Opt, Tour_FN_3Opt, Distance_FN_3Opt  = [], [], [], []
# for file in range(len(Coordinates)):
file = 5
## Record the tour initiated from 2-opt with nearest neighbor algorithm.
tour = Func_Three_Opt( Tour_NN_2Opt[ file ], distance_matrices[ file ])
Tour_NN_3Opt.append( tour[0] )
Distance_NN_3Opt.append( tour[1] )
print(file, ':NN')

## Record the tour initiated from 2-opt with farthest neighbor algorithm.
tour = Func_Three_Opt( Tour_FN_2Opt[ file ], distance_matrices[ file ])
Tour_FN_3Opt.append( tour[0] )
Distance_FN_3Opt.append( tour[1] )
print(file, ':FN')


Three_opt_NN, Three_opt_FN = [], []
# for file in range(len(Coordinates)):
temp = Distance_NN_3Opt[file][-1]
Three_opt_NN.append( temp )

temp = Distance_FN_3Opt[file][-1]
Three_opt_FN.append( temp )


# Distance_improved = pd.DataFrame({
#     'NN': Distance_NN,
#     'FN': Distance_FN,
#     '2_Opt_NN': Two_opt_NN,
#     '2_Opt_FN': Two_opt_FN,
#     '3_Opt_NN': Three_opt_NN,
#     '3_Opt_FN': Three_opt_FN,
#     }, index = [ name[:-4] for name in Data_FileNames ]
#     )




# %% Tabu Search

## Deteriorate the current solution by reversing a set of consecutive cities in the tour.
def Func_Deteriorate(tour):
    new_tour = tour.copy()
    n = len(new_tour) - 2  ## Exclude the starting node

    ## Select a set of consecutive cities with random length.
    # Ensure at least 5 cities can be reversed with maximum in 20% of cities.
    length_cities = int( max(5, n // 5) )
    start = random.randint(5, n - length_cities)  
    end = start + length_cities

    ## Reverse the segment of consecutive cities.
    new_tour[start:end] = new_tour[start:end][::-1]
    return new_tour


def Func_Tabu(tour, dist_matrix, tabu_list, stop_crit='iter_times', iter_times=100, tabu_length=1000, tabu_counts=5):
    n = len(tour)
    history = [ Func_Distance(tour, dist_matrix) ]
    temp_tabu_list = deepcopy(tabu_list)
    best_tours = []
    
    if stop_crit == 'iter_times':
        for t in range(iter_times):
            local_optimum = True
            for i in range(1, n - 2):  # Avoid the starting node
                for j in range(i + 1, n - 1):  # Ensure a valid swap
                    ## Check tabu list.
                    new_tour = deepcopy(tour)
                    new_tour[i:j+1] = new_tour[i:j+1][::-1]
                    if new_tour not in temp_tabu_list:
                        ## Calculate the distances of the current and proposed edges.
                        # If swapping reduces the distance, apply the change.
                        if Func_Distance(new_tour, dist_matrix) < Func_Distance(tour, dist_matrix):
                            tour = new_tour
                            temp_tabu_list.append( new_tour )
                            history.append( Func_Distance(tour, dist_matrix) )
                            best_tours.append( tour )
                            local_optimum = False
                            break
                if not local_optimum:
                    break
            
            ## When 2-opt no longer improve current solution, a local optimum is found.
            if local_optimum:
                tabu_list.append(tour)
                ## If a new better local optimum is found, clear the tabu list.
                if Func_Distance(tour, dist_matrix) < min([Func_Distance(tours, dist_matrix) for tours in tabu_list]):
                    temp_tabu_list = deepcopy(tabu_list)
                
                ## Jump out the local optimum to find the next optimum.
                tour = Func_Deteriorate(tour)
                
                
    elif stop_crit == 'tabu_length':
        while len(temp_tabu_list) < tabu_length:
            local_optimum = True
            for i in range(1, n - 2):  # Avoid the starting node
                for j in range(i + 1, n - 1):  # Ensure a valid swap
                    ## Check tabu list.
                    new_tour = deepcopy(tour)
                    new_tour[i:j+1] = new_tour[i:j+1][::-1]
                    if new_tour not in temp_tabu_list:
                        ## Calculate the distances of the current and proposed edges.
                        # If swapping reduces the distance, apply the change.
                        if Func_Distance(new_tour, dist_matrix) < Func_Distance(tour, dist_matrix):
                            tour = new_tour
                            temp_tabu_list.append( new_tour )
                            history.append( Func_Distance(tour, dist_matrix) )
                            best_tours.append( tour )
                            local_optimum = False
                            break
                if not local_optimum:
                    break
            
            ## When 2-opt no longer improve current solution, a local optimum is found.
            if local_optimum:
                tabu_list.append(tour)
                ## If a new better local optimum is found, clear the tabu list.
                if Func_Distance(tour, dist_matrix) < min([Func_Distance(tours, dist_matrix) for tours in tabu_list]):
                    temp_tabu_list = deepcopy(tabu_list)
                
                ## Jump out the local optimum to find the next optimum.
                tour = Func_Deteriorate(tour)
    
    
    elif stop_crit == 'tabu_counts':
        while len(tabu_list) < tabu_counts:
            local_optimum = True
            for i in range(1, n - 2):  # Avoid the starting node
                for j in range(i + 1, n - 1):  # Ensure a valid swap
                    ## Check tabu list.
                    new_tour = deepcopy(tour)
                    new_tour[i:j+1] = new_tour[i:j+1][::-1]
                    if new_tour not in temp_tabu_list:
                        ## Calculate the distances of the current and proposed edges.
                        # If swapping reduces the distance, apply the change.
                        if Func_Distance(new_tour, dist_matrix) < Func_Distance(tour, dist_matrix):
                            tour = new_tour
                            temp_tabu_list.append( new_tour )
                            history.append( Func_Distance(tour, dist_matrix) )
                            best_tours.append( tour )
                            local_optimum = False
                            break
                if not local_optimum:
                    break
            
            ## When 2-opt no longer improve current solution, a local optimum is found.
            if local_optimum:
                tabu_list.append(tour)
                ## If a new better local optimum is found, clear the tabu list.
                if Func_Distance(tour, dist_matrix) < min([Func_Distance(tours, dist_matrix) for tours in tabu_list]):
                    temp_tabu_list = deepcopy(tabu_list)
                
                ## Jump out the local optimum to find the next optimum.
                tour = Func_Deteriorate(tour)

    
    ## Find the best tour with minimum total distance.
    best_tour = best_tours[ history.index(min(history)) ]

    return best_tour, history
    
    
Tour_NN_Tabu, Distance_NN_Tabu, Tour_FN_Tabu, Distance_FN_Tabu  = [], [], [], []
# for file in range(len(Coordinates)):
file = 5
## Record the tour initiated from 2-opt with nearest neighbor algorithm.
tour = Func_Tabu( Tour_NN_2Opt[file], distance_matrices[file], 
                 [Tour_NN[file], Tour_FN[file], Tour_NN_2Opt[file], Tour_FN_2Opt[file]], 
                 stop_crit='tabu_length', tabu_length=1000 )
Tour_NN_Tabu.append( tour[0] )
Distance_NN_Tabu.append( tour[1] )
print(file, ':NN')

## Record the tour initiated from 2-opt with farthest neighbor algorithm.
tour = Func_Tabu( Tour_FN_2Opt[ file ], distance_matrices[ file ], 
                 [Tour_NN[file], Tour_FN[file], Tour_NN_2Opt[file], Tour_FN_2Opt[file]], 
                 stop_crit='tabu_length', tabu_length=1000 )
Tour_FN_Tabu.append( tour[0] )
Distance_FN_Tabu.append( tour[1] )
print(file, ':FN')


Tabu_NN = [ min(Distance_NN_Tabu[file]) for file in Distance_NN_Tabu[file] ]
Tabu_FN = [ min(Distance_FN_Tabu[file]) for file in Distance_FN_Tabu[file] ]



# Distance_improved = pd.DataFrame({
#     'NN': Distance_NN,
#     'FN': Distance_FN,
#     '2_Opt_NN': Two_opt_NN,
#     '2_Opt_FN': Two_opt_FN,
#     '3_Opt_NN': Three_opt_NN,
#     '3_Opt_FN': Three_opt_FN,
#     'Tabu_NN' : Tabu_NN,
#     'Tabu_FN' : Tabu_FN,
#     }, index = [ name[:-4] for name in Data_FileNames ]
#     )




# %% PSO

def Func_PSO(dist_matrix, num_particles=30, iter_times=100000, r_1 = 0.5, r_2 = 0.5, c_1 = 2, c_2 = 2, w = 0.7):
    num_particles = num_particles
    num_iter = iter_times
    r_1 = r_1   # prob of local best
    r_2 = r_2   # prob of global best
    c_1 = c_1   # coeff of local best
    c_2 = c_2   # coeff of global best
    w   = w     # coeff of inertia term

    Particles = []
    n = len(dist_matrix)
    ## Generate initial particles randomly.
    for i in range(num_particles):
        tour = list(range(n)) 
        tour.remove(start_node)
        random.shuffle(tour)
        tour = [start_node] + tour + [start_node]
        
        Particles.append(tour)


    Personal_bests = deepcopy(Particles)
    temp = [ Func_Distance(tours, dist_matrix) for tours in Particles ]
    Global_best = deepcopy( Particles[ temp.index(min(temp)) ] )
    History = []


    for t in range(num_iter):
        for i in range(num_particles):
            particle = deepcopy( Particles[i] )
            
            ## pbest update.
            if random.random() < r_1:
                length = int(1/c_1 * n)
                idx_start = random.randint(1, n-1 - length)  
                idx_end   = idx_start + length
                pbest = deepcopy( Personal_bests[i][idx_start: idx_end] )
            else:
                pbest = []
                
            ## gbest update.
            if random.random() < r_2:
                length = int(1/c_2 * n)
                idx_start = random.randint(1, n-1 - length)  
                idx_end   = idx_start + length
                gbest = deepcopy( Global_best[idx_start: idx_end] )
                gbest = [node for node in gbest if node not in pbest]
            else:
                gbest = []
                            
            ## inertia update.
            inertia = [node for node in particle[1:-1] if node not in (pbest + gbest)]
            length = int(w * len(inertia))
            if length > 0:
                idx_start = random.randint(1, len(inertia) - length)  
                idx_end   = idx_start + length
                inertia[idx_start:idx_end] = inertia[idx_start:idx_end][::-1]


            ## Construct a new particle.
            new_particle = [start_node] + pbest + gbest + inertia + [start_node]
            Particles[i] = deepcopy(new_particle)

            ## Update pbest and gbest.
            if Func_Distance(new_particle, dist_matrix) < Func_Distance(Personal_bests[i], dist_matrix):
                Personal_bests[i] = deepcopy(new_particle)
                
            if Func_Distance(new_particle, dist_matrix) < Func_Distance(Global_best, dist_matrix):
                Global_best = deepcopy(new_particle)
                History.append( (2*t+1, Func_Distance(Global_best, dist_matrix)) )

            
            ## For fairness, repeat the process by reversing the order of pbest and gbest update.
            particle = deepcopy( Particles[i] )
            
            ## gbest update.
            if random.random() < r_2:
                length = int(1/c_2 * n)
                idx_start = random.randint(1, n-1 - length)  
                idx_end   = idx_start + length
                gbest = deepcopy( Global_best[idx_start: idx_end] )
            else:
                gbest = []
                
            ## pbest update.
            if random.random() < r_1:
                length = int(1/c_1 * n)
                idx_start = random.randint(1, n-1 - length)  
                idx_end   = idx_start + length
                pbest = deepcopy( Personal_bests[i][idx_start: idx_end] )
                pbest = [node for node in pbest if node not in gbest]
            else:
                pbest = []
                            
            ## inertia update.
            inertia = [node for node in particle[1:-1] if node not in (pbest + gbest)]
            length = int(w * len(inertia))
            if length > 0:
                idx_start = random.randint(1, len(inertia) - length)  
                idx_end   = idx_start + length
                inertia[idx_start:idx_end] = inertia[idx_start:idx_end][::-1]


            ## Construct a new particle.
            new_particle = [start_node] + gbest + pbest + inertia + [start_node]
            Particles[i] = deepcopy(new_particle)

            ## Update pbest and gbest.
            if Func_Distance(new_particle, dist_matrix) < Func_Distance(Personal_bests[i], dist_matrix):
                Personal_bests[i] = deepcopy(new_particle)
                
            if Func_Distance(new_particle, dist_matrix) < Func_Distance(Global_best, dist_matrix):
                Global_best = deepcopy(new_particle)
                History.append( (2*t+2, Func_Distance(Global_best, dist_matrix)) )


    return Global_best, History


Tour_PSO, Cov_History_PSO  = [], []
# for file in range(len(Coordinates)):
for file in range(6, 10):
    # file = 5
    tour = Func_PSO( distance_matrices[file], 
                     num_particles=100, iter_times=5000, 
                     r_1 = 0.5, r_2 = 0.5, c_1 = 2, c_2 = 2, w = 0.7 )
    Tour_PSO.append( tour[0] )
    Cov_History_PSO.append( tour[1] )
    print(file, ':NN')
    

Dist_PSO = [ file[-1][1] for file in Cov_History_PSO ]




# Distance_improved = pd.DataFrame({
#     'NN': Distance_NN,
#     'FN': Distance_FN,
#     '2_Opt_NN': Two_opt_NN,
#     '2_Opt_FN': Two_opt_FN,
#     '3_Opt_NN': Three_opt_NN,
#     '3_Opt_FN': Three_opt_FN,
#     'Tabu_NN' : Tabu_NN,
#     'Tabu_FN' : Tabu_FN,
#     'PSO'     : Dist_PSO
#     }, index = [ name[:-4] for name in Data_FileNames ]
#     )




# %% Hybrid: PSO - 2-Opt Algorithm

def Func_Hybrid(dist_matrix, num_particles=30, iter_times=100000, r_1 = 0.5, r_2 = 0.5, c_1 = 2, c_2 = 2):
    num_particles = num_particles
    num_iter = iter_times
    r_1 = r_1   # prob of local best
    r_2 = r_2   # prob of global best
    c_1 = c_1   # coeff of local best
    c_2 = c_2   # coeff of global best

    Particles = []
    n = len(dist_matrix)
    ## Generate initial particles randomly.
    for i in range(num_particles):
        neighborhood = list(range(n))
        neighborhood.remove(start_node)
        random.shuffle(neighborhood)
        tour = [start_node] + deepcopy(neighborhood) + [start_node]
        
        Particles.append(tour)


    Personal_bests = deepcopy(Particles)
    temp = [ Func_Distance(tours, dist_matrix) for tours in Particles ]
    Global_best = deepcopy( Particles[ temp.index(min(temp)) ] )
    History = []
    ## Employ the Lin-Kernighan neighborhood as the inertia.
    LK_neighbor = random.choice(neighborhood)
    

    ## For the t-th exploration.
    for t in range(num_iter):
        ## For the i-th particle.
        for i in range(num_particles):
            particle = deepcopy( Particles[i] )
            
            ## pbest update.
            if random.random() < r_1:
                length = int(1/c_1 * n)
                idx_start = random.randint(1, n-1 - length)  
                idx_end   = idx_start + length
                pbest = deepcopy( Personal_bests[i][idx_start: idx_end] )
            else:
                pbest = []
                
            ## gbest update.
            if random.random() < r_2:
                length = int(1/c_2 * n)
                idx_start = random.randint(1, n-1 - length)  
                idx_end   = idx_start + length
                gbest = deepcopy( Global_best[idx_start: idx_end] )
                gbest = [node for node in gbest if node not in pbest]
            else:
                gbest = []
                            
            ## 2-Opt update.
            Opt_2 = [node for node in particle[1:-1] if node not in (pbest + gbest)]
            if len(Opt_2) > 0:
                Opt_2 = Func_Two_Opt(Opt_2, dist_matrix)[0]

            ## Construct a new particle.
            new_particle = [start_node] + pbest + gbest + Opt_2 + [start_node]
            ## inertia update: Lin-Kernighan neighborhood.
            # Check whether it is not the city after the starting node.
            idx_LK = new_particle.index(LK_neighbor)
            if idx_LK == 1:
                while True:
                    LK_neighbor = random.choice(neighborhood)
                    if LK_neighbor != new_particle[1]:  
                        break
                idx_LK = new_particle.index(LK_neighbor)

            new_particle[idx_LK], new_particle[idx_LK - 1] = \
                new_particle[idx_LK - 1], new_particle[idx_LK]
            Particles[i] = deepcopy(new_particle)

            ## Update pbest and gbest.
            if Func_Distance(new_particle, dist_matrix) < Func_Distance(Personal_bests[i], dist_matrix):
                Personal_bests[i] = deepcopy(new_particle)
                
            if Func_Distance(new_particle, dist_matrix) < Func_Distance(Global_best, dist_matrix):
                Global_best = deepcopy(new_particle)
                History.append( (2*t+1, Func_Distance(Global_best, dist_matrix)) )

            
            ## For fairness, repeat the process by reversing the order of pbest and gbest update.
            particle = deepcopy( Particles[i] )
            
            ## gbest update.
            if random.random() < r_2:
                length = int(1/c_2 * n)
                idx_start = random.randint(1, n-1 - length)  
                idx_end   = idx_start + length
                gbest = deepcopy( Global_best[idx_start: idx_end] )
            else:
                gbest = []
                
            ## pbest update.
            if random.random() < r_1:
                length = int(1/c_1 * n)
                idx_start = random.randint(1, n-1 - length)  
                idx_end   = idx_start + length
                pbest = deepcopy( Personal_bests[i][idx_start: idx_end] )
                pbest = [node for node in pbest if node not in gbest]
            else:
                pbest = []
                            
            ## 2-Opt update.
            Opt_2 = [node for node in particle[1:-1] if node not in (pbest + gbest)]
            if len(Opt_2) > 0:
                Opt_2 = Func_Two_Opt(Opt_2, dist_matrix)[0]

            ## Construct a new particle.
            new_particle = [start_node] + pbest + gbest + Opt_2 + [start_node]
            ## inertia update: Lin-Kernighan neighborhood.
            # Check whether it is not the city after the starting node.
            idx_LK = new_particle.index(LK_neighbor)
            if idx_LK == 1:
                while True:
                    LK_neighbor = random.choice(neighborhood)
                    if LK_neighbor != new_particle[1]:  
                        break
                idx_LK = new_particle.index(LK_neighbor)

            new_particle[idx_LK], new_particle[idx_LK - 1] = \
                new_particle[idx_LK - 1], new_particle[idx_LK]
            Particles[i] = deepcopy(new_particle)

            ## Update pbest and gbest.
            if Func_Distance(new_particle, dist_matrix) < Func_Distance(Personal_bests[i], dist_matrix):
                Personal_bests[i] = deepcopy(new_particle)
                
            if Func_Distance(new_particle, dist_matrix) < Func_Distance(Global_best, dist_matrix):
                Global_best = deepcopy(new_particle)
                History.append( (2*t+2, Func_Distance(Global_best, dist_matrix)) )


    return Global_best, History


Tour_Hybrid, Cov_History_Hybrid  = [], []
# for file in range(len(Coordinates)):
for file in [4, 6, 7, 8, 9]:
    # file = 5
    tour = Func_Hybrid( distance_matrices[file], 
                     num_particles=50, iter_times=5000, 
                     r_1 = 0.5, r_2 = 0.5, c_1 = 2, c_2 = 2 )
    Tour_Hybrid.append( tour[0] )
    Cov_History_Hybrid.append( tour[1] )
    print(file, ':Hybrid')
    

Dist_Hybrid = [ file[-1][1] for file in Cov_History_Hybrid ]








# %% Visualization: PSO

file = 3

x = [ cov[0] for cov in Cov_History_PSO[file] ]
y = [ cov[1] for cov in Cov_History_PSO[file] ]

plt.figure(figsize=(8, 5))
plt.plot(x, y, marker='x', linestyle='-', color='b')
plt.title("PSO Convergence History of " + Data_FileNames[file+6][:-4])
plt.xlabel("Number of Times of Convergence")
plt.ylabel("Total Distance of the Tour")
plt.grid(True)
plt.show()


Data_FileNames[file][:-4]




# %% Visualization: Hybrid

file = 0

x = [ cov[0] for cov in Cov_History_Hybrid[file] ]
y = [ cov[1] for cov in Cov_History_Hybrid[file] ]

plt.figure(figsize=(8, 5))
plt.plot(x, y, marker='x', linestyle='-', color='b')
plt.title("Hybrid: PSO-2-Opt Convergence History of " + Data_FileNames[file][:-4])
plt.xlabel("Number of Times of Convergence")
plt.ylabel("Total Distance of the Tour")
plt.grid(True)
plt.show()


Data_FileNames[file][:-4]




















