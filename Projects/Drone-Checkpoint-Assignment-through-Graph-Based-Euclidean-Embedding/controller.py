import warnings

import numpy as np
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import normalize


class Controller:
    def __init__(self, assignment_group, assignment_strategy, cse_strategy, cse_group=None):
        self.assignment_group = assignment_group
        self.assignment_strategy = assignment_strategy
        self.cse_strategy = cse_strategy
        self.cse_group = cse_group
        if not self.assignment_group == 'points':
            self.cluster_assignment = []

        self.prev_labels = None
        self.MI = []
        self.geometric = []

        self.robustness_scores = []
        self.prev_assignments = None

    def get_directions(self, world_object):
        """
        RETURNS THE DIRECTIONS IN WHICH EACH DRONE SHOULD PROGRESS
        :param world_object: object containing all world information
        :return: directions
        """
        self.set_assignment(world_object)
        target_coord = np.zeros((world_object.n_agents, 3))
        for i in range(world_object.n_agents):
            target_coord[i, :] = world_object.checkpoints_coord[world_object.agents_dict[i]['assigned_idx'], :]
        directions = target_coord - world_object.agents_coord
        directions = normalize(directions, axis=1)
        return directions

    def get_checkpoint_cluster_labels(self, D, assignment_group, n_clusters):
        if assignment_group == 'clustering':
            SC = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")

            try:
                labels = SC.fit_predict(D)
            except:
                """plt.imshow(D)
                plt.colorbar()
                plt.show()"""
                print('error')
                arr = []
                for i in range(int(D.shape[0] / n_clusters) + 1):
                    arr.append(np.arange(n_clusters))
                arr = np.concatenate(arr)
                labels = arr[n_clusters:]
                np.random.shuffle(labels)
            return labels

    def get_KMeans_labels(self, world_object, Y_embed=None):
        kmeans = KMeans(n_clusters=world_object.n_agents, n_init=1, precompute_distances=True, algorithm='elkan')
        if Y_embed is None:
            labels = kmeans.fit_predict(world_object.checkpoints_coord[world_object.remaining_checkpoints, :])
        else:
            labels = kmeans.fit_predict(Y_embed)
            if not len(np.unique(labels)) == world_object.n_agents:
                if np.max(np.unique(labels)) > len(np.unique(labels)) - 1:
                    unlabeled = []
                    for i in range(len(np.unique(labels))):
                        if i not in np.unique(labels):
                            unlabeled.append(i)
                    badlabeled = []
                    for i in np.unique(labels):
                        if i > len(np.unique(labels)) - 1:
                            badlabeled.append(i)
                    for i, bad in enumerate(badlabeled):
                        labels[np.nonzero(labels == bad)[0]] = unlabeled[i]
                unique_counts = []
                count = 0
                while True:
                    for i in np.unique(labels):
                        unique_counts.append(np.sum(labels == i))
                    n_splits = world_object.n_agents - len(np.unique(labels))
                    idxs = np.nonzero(np.argmax(unique_counts) == labels)
                    print(unique_counts)
                    split_idxs = np.array_split(idxs[0], n_splits + 1)
                    labels = np.array(labels)
                    for i, idx in enumerate(split_idxs):
                        if not i == 0:
                            labels[idx] = np.max(labels) + 1
                    labels = list(labels)

                    if len(np.unique(labels)) == world_object.n_agents:
                        break
                    if count == 5:
                        arr = []
                        for i in range(int(len(labels) / world_object.n_agents) + 1):
                            arr.append(np.arange(world_object.n_agents))
                        arr = np.concatenate(arr)
                        labels = arr[world_object.n_agents:]
                        np.random.shuffle(labels)

                        break

                    count += 1
        return labels

    def get_permute_idx(self, labels):
        perm = []
        for j in range(labels.max() + 1):
            for i in range(len(labels)):
                if labels[i] == j:
                    perm.append(i)
        return perm

    def get_permuted_matrix(self, matrix, perm):
        permuted_matrix = np.zeros(matrix.shape)
        permuted_matrix[:, :] = matrix[perm, :]
        permuted_matrix[:, :] = permuted_matrix[:, perm]
        return permuted_matrix

    def get_assignment_probabilities(self, X_agents, X_checkpoints):
        n = X_agents.shape[0]
        m = X_checkpoints.shape[0]

        l_u = (0, 1)
        A_eq = np.zeros((n, n * m))
        for i, x in enumerate(A_eq):
            for y in range(i * m, (i + 1) * m):
                A_eq[i, y] = 1
        b_eq = np.ones((n, 1))

        A_ineq = np.zeros((m, n * m))
        for i in range(m):
            for j in range(n):
                A_ineq[i, i + j * m] = 1
        b_ineq = np.ones((m, 1))
        if n > m:
            A_ineq = -A_ineq
            b_ineq = -b_ineq

        costs = np.linalg.norm(X_agents[:, np.newaxis, :] - X_checkpoints[np.newaxis, :, :], axis=2).flatten()[
                np.newaxis, :]

        P = linprog(costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ineq, b_ub=b_ineq, bounds=l_u, method='highs-ds').x
        P = np.reshape(P, (n, m))

        return P

    def get_assignments_euclidean(self, X_agents, X_checkpoints):
        """
        GET ASSIGNMENTS FOR EACH AGENT
        :param X_agents:
        :param X_checkpoints:
        :return:
        """
        if X_agents.shape[0] > 1:
            P = self.get_assignment_probabilities(X_agents, X_checkpoints)
            assignments = np.argmax(P, axis=1)
        else:
            distances = X_agents - X_checkpoints
            distances = np.linalg.norm(distances, axis=1)
            assignments = [np.argmin(distances)]
        return assignments

    def set_assignment(self, world_object):
        """
        RETURNS THE THE INDEX OF THE CHECKPOINTS TO WHICH THE DRONES ARE ASSIGNED
        :param agents_coord: coordinates of each drone
        :param checkpoints_coord: coordinates of each checkpoint
        :return: assignments
        """
        assignments = []
        if self.assignment_group == 'points' or world_object.n_agents >= len(world_object.remaining_checkpoints):

            if self.assignment_strategy == 'random':
                free_agents = 0
                for i in range(world_object.n_agents):
                    if world_object.agents_dict[i]['assigned_idx'] is None:
                        free_agents += 1
                if free_agents > 0:
                    for i in world_object.remaining_checkpoints:
                        assign = True
                        for j in range(world_object.n_agents):
                            if world_object.agents_dict[j]['assigned_idx']:
                                if i == world_object.agents_dict[j]['assigned_idx']:
                                    assign = False
                        if assign: assignments.append(i)
                    if len(assignments) > 0:
                        if free_agents > len(assignments):
                            for i in range(int(free_agents / len(assignments))):
                                assignments_ = assignments.copy()
                                np.random.shuffle(assignments_)
                                assignments = assignments + assignments_
                        assignments = assignments[:free_agents]
                        np.random.shuffle(assignments)
                        update = 0
                        for i in range(world_object.n_agents):
                            if world_object.agents_dict[i]['assigned_idx'] is None:
                                world_object.agents_dict[i]['assigned_idx'] = assignments[update]
                                update += 1
                    else:
                        for i in range(world_object.n_agents):
                            if world_object.agents_dict[i]['assigned_idx'] is None:
                                world_object.agents_dict[i]['assigned_idx'] = np.random.choice(
                                    world_object.remaining_checkpoints)

                # for i in range(int(world_object.agents_coord.shape[0]/len(world_object.remaining_checkpoints))+1):
                #     idx = np.arange(len(world_object.remaining_checkpoints))
                #     np.random.shuffle(idx)
                #     assignments = assignments + idx.tolist()
                # assignments = assignments[:world_object.agents_coord.shape[0]]
                # np.random.shuffle(assignments)
                # assignments = list(np.array(world_object.remaining_checkpoints)[assignments])

            elif self.assignment_strategy == 'CSE':
                checkpoints_unassigned = list(np.arange(len(world_object.remaining_checkpoints)))
                agents_unassigned = []
                do_embed = False
                for i in range(world_object.n_agents):
                    if world_object.agents_dict[i]['assigned_idx'] is None:
                        agents_unassigned.append(i)
                        do_embed = True
                    else:
                        checkpoints_unassigned.remove(
                            world_object.remaining_checkpoints.index(world_object.agents_dict[i]['assigned_idx']))

                if do_embed:
                    X_embed = world_object.get_CSE(
                        world_object.get_distance_matrix(world_object.get_adjacency_matrix()), 5)

                    X_embed_agents = X_embed[:world_object.n_agents, :]
                    X_embed_agents_unassigned = X_embed_agents[agents_unassigned, :]
                    X_embed_checkpoints = X_embed[world_object.n_agents:, :]
                    X_embed_checkpoints_unassigned = X_embed_checkpoints[checkpoints_unassigned, :]
                    assignments = self.get_assignments_euclidean(X_embed_agents_unassigned,
                                                                 X_embed_checkpoints_unassigned)

                    assignments = list(np.array(checkpoints_unassigned)[assignments])
                    assignments = list(np.array(world_object.remaining_checkpoints)[assignments])
                    for i, j in enumerate(agents_unassigned):
                        world_object.agents_dict[j]['assigned_idx'] = assignments[i]

            elif self.assignment_strategy == 'euclidean':
                checkpoints_unassigned = list(np.arange(len(world_object.remaining_checkpoints)))
                agents_unassigned = []
                do_assignment = False
                for i in range(world_object.n_agents):
                    if world_object.agents_dict[i]['assigned_idx'] is None:
                        agents_unassigned.append(i)
                        do_assignment = True
                    else:
                        checkpoints_unassigned.remove(
                            world_object.remaining_checkpoints.index(world_object.agents_dict[i]['assigned_idx']))

                if do_assignment:
                    X_agents_unassigned = world_object.agents_coord[agents_unassigned, :]
                    X_checkpoints_unassigned = world_object.checkpoints_coord[list(
                        np.array(world_object.remaining_checkpoints)[checkpoints_unassigned]), :]
                    assignments = self.get_assignments_euclidean(X_agents_unassigned,
                                                                 X_checkpoints_unassigned)
                    assignments = list(np.array(checkpoints_unassigned)[assignments])
                    assignments = list(np.array(world_object.remaining_checkpoints)[assignments])
                    for i, j in enumerate(agents_unassigned):
                        world_object.agents_dict[j]['assigned_idx'] = assignments[i]

            elif self.assignment_strategy == 'graph':
                # COUNTS NUMBER OF FREE AGENTS
                A = world_object.get_adjacency_matrix()
                for i in range(world_object.n_agents):
                    if world_object.agents_dict[i]['assigned_idx'] is None:
                        connections = np.nonzero(A[i, world_object.n_agents:] == 1)[0]
                        free_connections = []
                        if list(connections):
                            for connect in connections:
                                free = True
                                for j in range(world_object.n_agents):
                                    if world_object.agents_dict[i]['assigned_idx'] == \
                                            world_object.remaining_checkpoints[i]:
                                        free = False
                                        break
                                if free:
                                    free_connections.append(connect)
                            if free_connections:
                                idx = np.random.choice(free_connections)
                                world_object.agents_dict[i]['assigned_idx'] = world_object.remaining_checkpoints[idx]
                            else:
                                for checkpoint in world_object.remaining_checkpoints:
                                    free = True
                                    for j in range(world_object.n_agents):
                                        if world_object.agents_dict[i]['assigned_idx'] == \
                                                world_object.remaining_checkpoints[i]:
                                            free = False
                                            break
                                    if free:
                                        free_connections.append(checkpoint)
                                if free_connections:
                                    idx = np.random.choice(free_connections)
                                    world_object.agents_dict[i]['assigned_idx'] = world_object.remaining_checkpoints[
                                        idx]
                                else:
                                    world_object.agents_dict[i]['assigned_idx'] = world_object.remaining_checkpoints(
                                        np.random.choice(connections))
                        else:

                            for checkpoint in world_object.remaining_checkpoints:
                                free = True
                                for j in range(world_object.n_agents):
                                    if world_object.agents_dict[i]['assigned_idx'] == \
                                            world_object.remaining_checkpoints[i]:
                                        free = False
                                        break
                                if free:
                                    free_connections.append(checkpoint)
                            if free_connections:
                                idx = np.random.choice(free_connections)
                                world_object.agents_dict[i]['assigned_idx'] = idx
                            else:
                                world_object.agents_dict[i]['assigned_idx'] = np.random.choice(
                                    world_object.remaining_checkpoints)


        elif self.assignment_group == 'clustering':
            # STRATEGITES TO INVESTIGATE
            # 1: recluster each iter, reassign after reclsutering
            # 2: recluster only if clsuter is empty and reassign all drones to clusters and immediatley reassign drones to
            #    ckeckpoints in new clusters
            # 3: recluster only if clsuter is empty and reassign all drones to clusters. However, only reassign drone to new
            #    checkpoint if it reached previously assigned checkpoint. (catch: what if drone A assigned to cluster 1 is still
            #    assigned to only checkpoint in cluster 2? Then drone B has no free target. Best options: reassign drone A or have
            #    both drone A and B target checkoint in cluster 2. Latter is easiest to program.
            # 4: recluster if drone reached a checkpoint (when number of checkpoints changes) but only reassign drone that reached checkpoint
            recluster = False
            reassign = False
            if self.cse_strategy == 1:
                recluster = True
            else:
                for i in range(world_object.n_agents):
                    if not world_object.agents_dict[i]['cluster_idx'] and (
                            self.cse_strategy == 2 or self.cse_strategy == 3):
                        recluster = True
                        break
                    elif self.cse_strategy == 4 and world_object.agents_dict[i]['assigned_idx'] is None:
                        recluster = True
                        break
                    elif world_object.agents_dict[i]['assigned_idx'] is None:
                        reassign = True

            if (recluster or reassign):
                if self.assignment_strategy == 'CSE' or self.assignment_strategy == 'random':
                    D = world_object.get_distance_matrix(world_object.get_adjacency_matrix())
                if self.assignment_strategy == 'CSE':
                    X_embed = world_object.get_CSE(D, 5)
                    X_embed_agents = X_embed[:world_object.n_agents, :]
                    X_embed_checkpoints = X_embed[world_object.n_agents:, :]

            if recluster:
                if self.assignment_strategy == 'random' or self.assignment_strategy == 'CSE':
                    D_copy = D.copy()
                    D_copy = np.nan_to_num(D_copy, posinf=0)
                    D = np.nan_to_num(D, posinf=D_copy.max() * 6)
                    D_checkpoints = D[world_object.n_agents:, world_object.n_agents:]
                    if self.assignment_strategy == 'random' or self.cse_group == 'spectral':
                        labels = self.get_checkpoint_cluster_labels(D_checkpoints,
                                                                    assignment_group=self.assignment_group,
                                                                    n_clusters=world_object.n_agents)
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            labels = self.get_KMeans_labels(world_object=world_object, Y_embed=X_embed_checkpoints)
                    check_assignment = False
                    if not self.prev_labels is None and len(self.prev_labels) == len(labels):
                        self.MI.append(normalized_mutual_info_score(self.prev_labels, labels))
                        self.geometric.append(fowlkes_mallows_score(self.prev_labels, labels))
                        check_assignment = True

                    self.prev_labels = labels

                    # perm = self.get_permute_idx(labels_SC)
                    # permuted_D = self.get_permuted_matrix(D_checkpoints, perm)
                    # world_object.save_distance_plot(world_object.get_distance_matrix(world_object.get_adjacency_matrix())[world_object.n_agents:, world_object.n_agents:])
                    # idx = np.nonzero(permuted_D > D_copy.max())
                    # permuted_D[idx]=np.inf
                    # world_object.save_distance_plot(permuted_D, boundaries='clusters', cluster_labels=labels_SC)# DISPLAY CLUSTERING

                    if self.assignment_strategy == 'random':
                        idx = np.arange(world_object.n_agents)
                        np.random.shuffle(idx)

                        for i in np.unique(labels):
                            class_idx = list(np.nonzero(labels == i)[0])
                            world_object.agents_dict[idx[i]]['cluster_idx'] = list(
                                np.array(world_object.remaining_checkpoints)[class_idx])

                    elif self.assignment_strategy == 'CSE':
                        cluster_centers = np.zeros((world_object.n_agents, X_embed.shape[1]))
                        class_idxs = []
                        for i in np.unique(labels):
                            class_idx = list(np.nonzero(labels == i)[0])
                            class_idxs.append(class_idx)
                            X_embed_checkpoints_i = X_embed_checkpoints[class_idx, :]
                            cluster_centers[i, :] = np.sum(X_embed_checkpoints_i, axis=0) / X_embed_checkpoints_i.shape[
                                0]

                        cluster_assignments = self.get_assignments_euclidean(X_embed_agents, cluster_centers)
                        for i in range(world_object.n_agents):
                            world_object.agents_dict[i]['cluster_idx'] = list(
                                np.array(world_object.remaining_checkpoints)[class_idxs[cluster_assignments[i]]])
                elif self.assignment_strategy == 'euclidean':
                    labels_KMeans = self.get_KMeans_labels(world_object=world_object)
                    check_assignment = False
                    if not self.prev_labels is None and len(self.prev_labels) == len(labels_KMeans):
                        self.MI.append(normalized_mutual_info_score(self.prev_labels, labels_KMeans))
                        self.geometric.append(fowlkes_mallows_score(self.prev_labels, labels_KMeans))
                        check_assignment = True
                        print(fowlkes_mallows_score(self.prev_labels, labels_KMeans))

                    self.prev_labels = labels_KMeans

                    cluster_centers = np.zeros((world_object.n_agents, 3))
                    class_idxs = []
                    for i in np.unique(labels_KMeans):
                        class_idx = list(np.nonzero(labels_KMeans == i)[0])
                        class_idxs.append(class_idx)
                        X_checkpoints_i = world_object.checkpoints_coord[
                                          list(np.array(world_object.remaining_checkpoints)[class_idx]), :]
                        cluster_centers[i, :] = np.sum(X_checkpoints_i, axis=0) / X_checkpoints_i.shape[0]

                    cluster_assignments = self.get_assignments_euclidean(world_object.agents_coord, cluster_centers)

                    for i in range(world_object.n_agents):
                        world_object.agents_dict[i]['cluster_idx'] = list(
                            np.array(world_object.remaining_checkpoints)[class_idxs[cluster_assignments[i]]])

            if self.assignment_strategy == 'random':
                for i in range(world_object.n_agents):
                    if not world_object.agents_dict[i]['assigned_idx']:
                        assignments.append(np.random.choice(world_object.agents_dict[i]['cluster_idx']))
                    else:
                        assignments.append(world_object.agents_dict[i]['assigned_idx'])
                for i in range(world_object.n_agents):
                    world_object.agents_dict[i]['assigned_idx'] = assignments[i]
            elif self.assignment_strategy == 'euclidean':
                assignments = []
                for i in range(world_object.n_agents):
                    if self.cse_strategy == 1 or (not world_object.agents_dict[i]['assigned_idx']) or (
                            self.cse_strategy == 2 and recluster):
                        distances = world_object.agents_coord[i][np.newaxis, :] - world_object.checkpoints_coord[
                            world_object.agents_dict[i]['cluster_idx']]
                        distances = np.linalg.norm(distances, axis=1)
                        assignment = np.argmin(distances)
                        assignment = world_object.agents_dict[i]['cluster_idx'][assignment]
                        assignments.append(assignment)
                        world_object.agents_dict[i]['assigned_idx'] = assignment
                    if check_assignment:
                        self.robustness_scores.append(
                            np.sum(np.array(self.prev_assignments) == np.array(assignments)) / len(
                                self.prev_assignments))
                    self.prev_assignments = assignments
            elif self.assignment_strategy == 'CSE':
                if recluster or reassign:
                    assignments = []
                    for i in range(world_object.n_agents):
                        if self.cse_strategy == 1 or (not world_object.agents_dict[i]['assigned_idx']) or (
                                self.cse_strategy == 2 and recluster):
                            cluster_idx = []

                            for idx in world_object.agents_dict[i]['cluster_idx']:
                                cluster_idx.append(world_object.remaining_checkpoints.index(idx))

                            assignment = self.get_assignments_euclidean(X_embed_agents[i, :][np.newaxis, :],
                                                                        X_embed_checkpoints[cluster_idx, :])

                            assignment = world_object.agents_dict[i]['cluster_idx'][assignment[0]]
                            assignments.append(assignment)

                            world_object.agents_dict[i]['assigned_idx'] = assignment
                    if check_assignment:
                        self.robustness_scores.append(
                            np.sum(np.array(self.prev_assignments) == np.array(assignments)) / len(
                                self.prev_assignments))
                    self.prev_assignments = assignments
