import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh


class World:
    def __init__(self, n_agents, n_checkpoints, world_size, view_radius, random_seed=None):
        """
        INITIATES THE WORLD PARAMETERS
        :param n_agents (int): Number of drones in our world
        :param n_checkpoints (int): Number of checkpoints at t=0
        :param world_size (float list): [x,y,z] dimensions of the world in metres
        :param world_radius (float): radius of view of drones in metres
        """
        self.n_agents = n_agents
        self.n_checkpoints = n_checkpoints
        self.world_size = world_size
        self.view_radius = view_radius
        self.generate_world(random_seed=random_seed)

    def generate_world(self, random_seed):
        """
        CREATES COORDINATES OF DRONES AND CHECKPOINTS
        :param random_seed: set in case we want determninistic results
        :return: No return
        """
        if random_seed is not None: np.random.seed(random_seed)

        self.agents_coord = np.random.uniform(0, 1, (self.n_agents, 3))
        self.agents_coord = self.agents_coord * np.array(self.world_size)[np.newaxis, :]
        self.agents_dict = {}
        for i in range(self.n_agents):
            self.agents_dict[i] = {'cluster_idx': None, 'assigned_idx': None}

        self.checkpoints_coord = np.random.uniform(0, 1, (self.n_checkpoints, 3))
        self.checkpoints_coord = self.checkpoints_coord * np.array(self.world_size)[np.newaxis, :]

        self.remaining_checkpoints = []
        for i in range(self.checkpoints_coord.shape[0]):
            self.remaining_checkpoints.append(i)

    def get_adjacency_matrix(self):
        """
        RETURNS ADJACENCY MATRIX OF ENTIRE WORLD
        :return: A (n_agents+n_checkpoints, n_agents+n_checkpoints): adjacency matrix of world
        """
        all_coord = np.concatenate((self.agents_coord, self.checkpoints_coord[self.remaining_checkpoints]), axis=0)
        A = np.eye(self.n_agents + len(self.remaining_checkpoints), dtype=int)
        check = []
        for i in range(self.n_agents):
            check.append(i)
            for j in range(self.n_agents + len(self.remaining_checkpoints)):
                if np.linalg.norm(all_coord[i, :] - all_coord[j, :]) <= self.view_radius:
                    A[i, j] = 1
                    A[j, i] = 1
        return A

    def get_distance_matrix(self, adjacency_matrix, method=None):
        """
        CALCULATES DISTANCE MATRIX OF ADJACENCY MATRIX
        :param adjacency_matrix (n_agents+n_checkpoints, n_agents+n_checkpoints)
        :return: D (n_agents+n_checkpoints, n_agents+n_checkpoints): distance matrix of world
        """
        if method == None: method = 'auto'
        D = shortest_path(csr_matrix(adjacency_matrix), directed=False, unweighted=True, overwrite=False, indices=None,
                          method=method)
        return D

    def save_graph(self, adjacency_matrix):
        colors = np.zeros((self.n_agents + self.checkpoints_coord.shape[0]))
        colors[:self.n_agents] = 1
        gr = nx.convert_matrix.from_numpy_matrix(adjacency_matrix)

        nx.draw(gr, node_size=50, node_color=colors, width=0.5)  # ,labels=mylabels)
        plt.savefig('graph.png', dpi=200)
        plt.close()

    def save_adjacency_plot(self, A, cmap='binary', alpha=0.3):
        plt.imshow(A, cmap=cmap, extent=(
        0, self.n_agents + self.checkpoints_coord.shape[0], self.n_agents + self.checkpoints_coord.shape[0], 0))
        plt.colorbar()
        plt.axhspan(xmin=0, xmax=((self.n_agents) / (self.n_agents + self.checkpoints_coord.shape[0])), ymin=0,
                    ymax=((self.n_agents)), alpha=alpha, facecolor='g')
        plt.axhspan(xmin=((self.n_agents) / (self.n_agents + self.checkpoints_coord.shape[0])), xmax=1,
                    ymin=((self.n_agents)), ymax=self.n_agents + self.checkpoints_coord.shape[0], alpha=alpha,
                    facecolor='r')
        plt.legend(handles=[Patch(facecolor='g', alpha=0.3, label='Edges between agents'),
                            Patch(facecolor='r', alpha=0.3, label='Edges between checkpoints')],
                   bbox_to_anchor=(0, 1.16), loc='upper left')
        plt.savefig('adjacency.png', dpi=200)
        plt.close()

    def save_distance_plot(self, D, cmap='copper', boundaries=None, cluster_labels=None):
        if boundaries == 'agents_and_checkpoints':
            plt.imshow(D, cmap=cmap, extent=(0, D.shape[0], D.shape[0], 0))
            plt.colorbar()
            plt.axhspan(xmin=0, xmax=((self.n_agents) / (self.n_agents + self.checkpoints_coord.shape[0])), ymin=0,
                        ymax=((self.n_agents)), facecolor='none', edgecolor='g', linewidth=2)
            plt.axhspan(xmin=((self.n_agents) / (self.n_agents + self.checkpoints_coord.shape[0])), xmax=1,
                        ymin=((self.n_agents)), ymax=self.n_agents + self.checkpoints_coord.shape[0], facecolor='none',
                        edgecolor='r', linewidth=2)
            plt.legend(
                handles=[Patch(facecolor='none', edgecolor='g', label='Shortest path between agents', linewidth=2),
                         Patch(facecolor='none', edgecolor='r', label='Shortest path between checkpoints',
                               linewidth=2)], bbox_to_anchor=(0, 1.16), loc='upper left')
            plt.savefig('distance_agents_and_checkpoints.png', dpi=200)
            plt.close()
        elif boundaries == 'clusters':
            plt.imshow(D, cmap=cmap, extent=(0, D.shape[0], D.shape[0], 0))
            plt.colorbar()

            count = 0
            for i in np.unique(cluster_labels):
                interm_count = count
                count += np.sum(cluster_labels == i)
                plt.axhspan(xmin=interm_count / (D.shape[0]), xmax=count / (D.shape[0]), ymin=interm_count, ymax=count,
                            facecolor='none', edgecolor='r', linewidth=2.5)
            plt.savefig('distance_clusters.png', dpi=200)
            plt.close()
        else:
            plt.imshow(D, cmap=cmap, extent=(0, D.shape[0], D.shape[0], 0))
            plt.colorbar()
            plt.savefig('distance.png', dpi=200)
            plt.close()

    def purge_checkpoints(self, reached_radius, assignment_group=None, purge_list=None):
        """
        REMOVES CHECKPOINTS THAT HAVE BEEN REACHED
        :param reached_radius: distance between drone and checkpoint when checkpoint is considered reached
        """
        do = False
        if purge_list is None:
            purge_list = []
            for i in range(self.n_agents):
                if np.linalg.norm(self.agents_coord[i, :] - self.checkpoints_coord[self.agents_dict[i]['assigned_idx'],
                                                            :]) < reached_radius:
                    purge_list.append(self.agents_dict[i]['assigned_idx'])

        for i in range(self.n_agents):
            if assignment_group == 'clustering': self.agents_dict[i]['cluster_idx'] = [checkpoint for checkpoint in
                                                                                       self.agents_dict[i][
                                                                                           'cluster_idx'] if
                                                                                       checkpoint not in purge_list]
            if self.agents_dict[i]['assigned_idx'] in purge_list:
                self.agents_dict[i]['assigned_idx'] = None

        self.remaining_checkpoints = [checkpoint for checkpoint in self.remaining_checkpoints if
                                      checkpoint not in purge_list]
        return purge_list

    def get_CSE(self, D, dim, return_Sc=False):
        """
        RETURNS EMBEDDINGS OF EACH POINT IN THE WORLD ACCORDING TO THE DISTANCE MATRIX
        :param D (n_agents, number of remaining agents): distance matrix of the system
        :return: X_embed (n_agents + number of remaining agents, embedding_dim): embedding of world
        """

        n = D.shape[0]
        D_copy = D.copy()
        D_copy = np.nan_to_num(D_copy, posinf=0)
        D = np.nan_to_num(D, posinf=D_copy.max() * 15)

        Q = np.eye(n) - (1 / n) * np.ones((n, n))
        Dc = Q @ D @ Q
        Sc = -(Dc + Dc.T) / 4  # forcing symmetrx

        lambda_nSc = eigsh(Sc, return_eigenvectors=False, which='SA', k=1).item()

        D_tilde = D - 2 * lambda_nSc * (np.ones((n, n)) - np.eye(n))

        Sc_tilde = Q @ D_tilde @ Q
        Sc_tilde = -(Sc_tilde + Sc_tilde.T) / 4

        e, v = eigh(Sc_tilde)

        try:
            n_zeros = np.max(np.where(e < 1e-3))
        except:
            n_zeros = -1
        e = np.flip(e[n_zeros + 1:])
        V = np.flip(v[:, n_zeros + 1:], axis=1)
        L = np.diag(e)

        X_embed = V[:, :dim] @ (L[:dim, :dim] ** 0.5)
        if return_Sc: return X_embed, Sc
        return X_embed
