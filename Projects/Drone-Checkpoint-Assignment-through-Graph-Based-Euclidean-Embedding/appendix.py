import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

from time import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from controller import Controller
from simulation import Simulation
from world import World


# APPENDIX A
n_agents = 10
n_checkpoints = 50
world_size = [1000,1000,1000]
view_radius = 600

assignment_group = 'clustering' #'clustering', 'points'
cse_group = 'kmeans'
assignment_strategy_list = ['CSE', 'euclidean']

cse_strategy = 1
speed = 10
dt = 1
finish_process = False


reevaluate = False

if reevaluate:
    for assignment_strategy in assignment_strategy_list:
        world = World(n_agents=n_agents, n_checkpoints=n_checkpoints, world_size=world_size,
                      view_radius=view_radius)
        controller = Controller(assignment_group=assignment_group,
                                assignment_strategy=assignment_strategy,
                                cse_strategy=cse_strategy, cse_group=cse_group)
        simulation = Simulation(speed=speed, dt=dt, finish_process=finish_process)
        print('in')
        simulation.simulate(world, controller, stop_after=10000)

        geometric_scores = np.array(controller.geometric)
        MI_scores = np.array(controller.geometric)
        robustness_scores = np.array(controller.robustness_scores)
        print(MI_scores)
        np.save(assignment_strategy+'_geometric_scores.npy', geometric_scores)
        np.save(assignment_strategy+'_MI_scores.npy', MI_scores)
        np.save(assignment_strategy+'_robustness_scores.npy', robustness_scores)

sumgeoms_list = []
sumMIs_list = []
sumRob_list = []
div=100
line_type='-'
fig=plt.figure()

ax1=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)
fig.set_size_inches(14,4)
for assignment_strategy in assignment_strategy_list:
    geometric_scores = np.load(assignment_strategy+'_geometric_scores.npy')
    MI_scores = np.load(assignment_strategy+'_MI_scores.npy')
    robustness_scores = np.load(assignment_strategy+'_robustness_scores.npy')
    sumgeom=0
    sumMI=0
    sumRob=0
    sumgeoms=[]
    sumMIs=[]
    sumRobs=[]
    for i in range(geometric_scores.shape[0]):
        if i%div==0 and i >0:
            sumgeoms.append(sumgeom/div)
            sumMIs.append(sumMI/div)
            sumRobs.append(sumRob/div)
            sumgeom=0
            sumMI=0
            sumRob=0
        sumgeom+=geometric_scores[i]
        sumMI+=MI_scores[i]
        sumRob+=robustness_scores[i]
    div=1
    #plt.plot(sumgeoms, label=assignment_strategy+'_geom', line_type, 'b')
    if assignment_strategy == 'CSE':
        print('in')
        ax1.plot(100*np.arange(len(sumMIs)), sumMIs, color='red', label='CSE method clustering mutual information scores')
        ax1.plot(100*np.arange(len(sumRobs)), sumRobs, color='red', linestyle='dashed', label='CSE method drone-checkpoint assignment scores')
        ax1.set_xlabel("Iter of CSE-based scheme", color="red")

        ax1.set_yticklabels([])
        ax1.get_yaxis().set_visible(False)
        ax1.legend(loc='lower left')


    elif assignment_strategy == 'euclidean':
        print(sumMIs)
        print(sumRobs)
        ax2.plot(np.arange(len(sumMIs)), sumMIs, color='green', label='Upper bound method clustering mutual information scores')
        ax2.plot(np.arange(len(sumRobs)), sumRobs, color='green', linestyle='dashed', label='Upper bound method drone-checkpoint assignment scores')
        ax2.set_xlabel("Iter of upper-bound-scheme", color="green")
        ax2.set_ylabel("Robustness Score")
        ax2.xaxis.tick_top()
        ax2.legend(loc='lower right')


        ax2.xaxis.set_label_position('top')

    line_type='--'
plt.savefig('robustness.png')

# APPENDIX B

world = World(n_agents=100, n_checkpoints=1500, world_size=world_size,
                      view_radius=view_radius)
_, Sc = world.get_CSE(world.get_distance_matrix(world.get_adjacency_matrix()), 1500, return_Sc=True)


from scipy.linalg import eigh
eigen_spectrum = np.flip(eigh(Sc, eigvals_only = True))

fig, axs = plt.subplots(1,2)
p_opt=5

try:n_zeros = np.min(np.where(eigen_spectrum<0))
except:print("no eigenvalues below 0")

fig.set_size_inches(14,4)
axs[0].plot(eigen_spectrum[:n_zeros], 'y', linewidth=4,label='Eigen Spectrum')
axs[0].set_title('Entire Spectrum')
v = np.linspace(eigen_spectrum[:n_zeros].min(),eigen_spectrum[:n_zeros].max(), num=len(eigen_spectrum[:n_zeros]))
vert=np.ones((len(eigen_spectrum[:n_zeros])))*p_opt
axs[0].plot(vert,v, 'b--', label='cut-off')
axs[0].legend()

range_ = np.linspace(0, 100, eigen_spectrum[:100].size)

axs[1].plot(range_, eigen_spectrum[:100], 'y', linewidth=4,label='Eigen Spectrum')
axs[1].set_title('Spectrum zoomed in on cut-off')
v = np.linspace(eigen_spectrum[:100].min(),eigen_spectrum[:100].max(), num=len(eigen_spectrum[:100]))
vert=np.ones((len(eigen_spectrum[:100])))*p_opt
axs[1].plot(vert,v, 'b--', label='cut-off')
axs[1].legend()
plt.savefig('eigenspectrum.png')

# APPENDIX C

linprog_times = []
linprog_in_cluster_times = []
linprog_cluster_assignment_times = []
CSE_times = []
clustering_times = []
graph_times=[]
n_agents=5
assignment_strategy='CSE'
reevaluate = False
if reevaluate:
    for n_checkpoints in tqdm(np.flip(np.arange(1500))):
        if n_agents==n_checkpoints:
            break

        world = World(n_agents=n_agents, n_checkpoints=n_checkpoints, world_size=world_size, view_radius=view_radius)
        start = time()
        A = world.get_adjacency_matrix()
        D = world.get_distance_matrix(A)
        graph_times.append(time()-start)

        start = time()
        X = world.get_CSE(D, dim=5)
        CSE_times.append(time()-start)

        controller = Controller(assignment_group=assignment_group, assignment_strategy=assignment_strategy, cse_strategy=3)

        start = time()
        assignments = controller.get_assignments_euclidean(X_agents=X[:5, :], X_checkpoints=X[5:, :])
        linprog_times.append(time()-start)

        start = time()
        distances = X[0, :][np.newaxis, :] - X[5:5+int(n_checkpoints/5), :][np.newaxis, :]
        distances = np.linalg.norm(distances, axis=1)
        assignment = np.argmin(distances)
        linprog_in_cluster_times.append(time()-start)

        start = time()
        centroids = np.random.rand(5, 5)
        assignment = controller.get_assignments_euclidean(X_agents=X[:5, :], X_checkpoints=centroids)
        linprog_cluster_assignment_times.append(time()-start)

        start = time()
        labels = controller.get_KMeans_labels(world_object=world, Y_embed=X[5:, :])
        clustering_times.append(time()-start)

    np.save('linprog_times.npy', np.array(linprog_times))
    np.save('linprog_in_cluster_times.npy', np.array(linprog_in_cluster_times))
    np.save('linprog_cluster_assignment_times.npy', np.array(linprog_cluster_assignment_times))
    np.save('CSE_times.npy', np.array(CSE_times))
    np.save('clustering_times.npy',np.array(clustering_times))
    np.save('graph_times.npy', np.array(graph_times))

linprog_times = np.load('linprog_times.npy')
linprog_in_cluster_times = np.load('linprog_in_cluster_times.npy')
linprog_cluster_assignment_times = np.load('linprog_cluster_assignment_times.npy')
CSE_times = np.load('CSE_times.npy')
clustering_times = np.load('clustering_times.npy',)
graph_times = np.load('graph_times.npy')

linprog=0
linprog_alt=0
CSE=0
graph=0
linprog_sums=[]
linprog_alt_sums=[]
CSE_sums=[]
graph_sums=[]
div=10
for i in range(linprog_times.shape[0]):
    if i%div==0 and i >0:
        linprog_sums.append(linprog/div)
        linprog_alt_sums.append(linprog_alt/div)
        CSE_sums.append(CSE/div)
        graph_sums.append(graph/div)
        linprog=0
        linprog_alt=0
        CSE=0
        graph=0
    linprog+=linprog_times[i]
    linprog_alt+=linprog_in_cluster_times[i]+linprog_cluster_assignment_times[i]+clustering_times[i]
    CSE+=CSE_times[i]
    graph+=graph_times[i]
plt.figure(figsize=(14, 4))

plt.plot(10*np.flip(np.arange(len(linprog_sums))), linprog_sums,label='Optimization problem (4)')
plt.plot(10*np.flip(np.arange(len(linprog_sums))), linprog_alt_sums, label='Clustering and optimization problems (9) and (10)')
# plt.plot(linprog_cluster_assignment_times)
plt.plot(10*np.flip(np.arange(len(linprog_sums))),CSE_sums,label='CSE')
#plt.plot(clustering_times)
plt.plot(10*np.flip(np.arange(len(linprog_sums))),graph_sums,label='Shortest distance matrix calculation')
plt.legend()
plt.xlabel("Number of checkpoints")
plt.ylabel("Processing time (s)")
plt.savefig('computational_analysis.png')
