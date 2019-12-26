import numpy as np
from sklearn.cluster import KMeans
import time
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix, lil_matrix, spmatrix
from scipy.sparse import csgraph
import gc


def form_graph(filename):
    '''
    form a graph from the .txt file
    :param file: data file
    :return: graph, in the shape used latter
            n, k
    '''
    a = time.time()
    with open('../data/{}'.format(filename), 'r') as f:
        first_line = f.readline()[:-1]  # remove '\n' at the end
        meta = first_line.split(' ')
        n, e, k = int(meta[2]), int(meta[3]), int(meta[-1])

        lines = f.readlines()

    graph = np.ndarray((e, 2), dtype=np.int64)
    for i, edge in enumerate(lines):
        s, t = edge[:-1].split(' ')
        graph[i] = int(s), int(t)

    del lines
    gc.collect()
    e = time.time()
    print('Time for form graph:{}'.format(e - a))
    return graph, n, k, first_line


def generate_adj(graph, n):
    '''
    generate the adjacency matrix of a graph
    :param graph: the edges of a graph
    :param n: the number of vertices in this graph
    :return: adjacency matrix
    '''
    a = time.time()
    adj = lil_matrix((n, n), dtype=np.int16)
    # adj = np.zeros((n, n))
    for s, t in graph:
        adj[s, t] = adj[t, s] = 1
    t = time.time()
    del graph
    gc.collect()
    print('Time for generate adjacency matrix:{}'.format(t - a))
    return adj


def generate_dia(adj, n):
    '''
    From adjacency matrix build diagonal matrix
    :param adj: adjacency matrix, a ndarray
    :param n: the number of vertices in this graph
    :return: diagonal matrix
    '''
    s = time.time()
    dia = csr_matrix((n, n), dtype=np.int16)
    # dia = np.zeros((n, n), dtype=np.int16)
    for i, r in enumerate(adj):
        dia[i][i] = sum(row)
    t = time.time()
    print('Time for generate diagonal matrix:{}'.format(t - s))
    return dia


def generate_lap(adj):
    '''
    From adjacency matrix and diagonal matrix build Laplacian matrix
    :param dia: diagonal matrix
    :param adj: adjacency matrix
    :return: Laplacian matrix
    '''
    s = time.time()
    # lap = dia - adj
    # # normalize lap
    # x = np.linalg.norm(lap)
    # lap = lap / x
    #
    # del dia
    # del adj
    lap = csgraph.laplacian(adj, normed=False)
    t = time.time()
    # gc.collect()
    print('Time for generate laplacian matrix:{}'.format(t - s))
    return lap


def compute_k_eigenvectors(lap, k):
    '''compute the first k eigenvectors of laplacian matrix
    :param lap: laplacian matrix
    :param k: a number
    :return: The normalized (first k) eigenvectors
    '''
    _, vectors = np.linalg.eig(lap)
    vectors = vectors.real

    return vectors[:k]


def get_U(lap, k):
    '''
    Using scipy.sparse.linalg.eigs to calculate matrix U that we need for kmeans algorithm
    :param lap: laplacian matrix
    :param k: a number
    :return: matrix U
    '''
    s = time.time()
    lap = lap.astype('float64')
    _, first_k = eigs(lap, k, sigma=0)
    U = first_k.real
    # normalize U
    x = np.linalg.norm(U)
    U = U / x
    t = time.time()
    del lap
    gc.collect()
    print('Time for get U:{}'.format(t - s))
    return U


def generate_u(vec_k):
    '''
    from first k vectors generate matrix U
    :param vec_k: first k eigenvectors
    :return: matrix U, using rows of vec_k as columns
    '''
    u = vec_k.T
    x = np.linalg.norm(u)
    u = u / x
    return u


def k_means(data, k):
    '''
    Using K-means algorithm to cluster the data
    :param data: n points
    :param k: number of clusters
    :return: clusters
    '''
    s = time.time()
    kmeans = KMeans(n_clusters=k, algorithm='auto')
    kmeans.fit(data)
    del data
    gc.collect()
    t = time.time()
    print('Time for run K-means algorithm:{}'.format(t - s))
    return kmeans.labels_


def get_clusters(labels, k, filename, firstline):
    '''
    return the clusters of vertices
    :param labels: labels generated from kmeans method
    :return: clusters
    '''
    s = time.time()
    clusters = [set() for _ in range(k)]
    with open('../results/{}_output.txt'.format(filename[:-4]), 'w') as f:
        f.write('{}\n'.format(firstline))
        for i, l in enumerate(labels):
            clusters[l].add(i)
            f.write('{} {}\n'.format(i, l))
    t = time.time()
    print('Time for writing result:{}'.format(t - s))
    return clusters



def partitioning_1(filename):
    s = time.time()
    graph, n, k, fl = form_graph(filename)
    adj = generate_adj(graph, n)
    # dia = generate_dia(adj, n)
    lap = generate_lap(adj)
    data = get_U(lap, k)
    labels = k_means(data, k)
    clusters = get_clusters(labels, k, filename, fl)
    t = time.time()
    print('Total time consumption:{}'.format(t - s))
    return clusters


if __name__ == '__main__':
    labels = partitioning_1('Oregon-1.txt')
    for i in labels:
        print(i)
        print(len(i))
