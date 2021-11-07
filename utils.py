from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power


def select_gygantic_component(G: nx.Graph) -> nx.Graph:
    """ 
    removes all DELETED profiles
    returns the largest connected component
    """
    g = G.copy()
    to_delete = []

    for n in g.nodes:
        if g.nodes[n]['first_name'] == 'DELETED':
            to_delete.append(n)

            # drop DELETED profiles
    g.remove_nodes_from(to_delete)

    largest_cc = max(nx.connected_components(g), key=len)

    return G.subgraph(largest_cc)


def calc_centralities(graph: nx.Graph) -> dict:
    """
    Calculate degree, closeness, betweenness centralities of the graph

    :param G: nx.Graph
    :return: mapping of centrality names (degree, closeness, betweenness) to np.array of its values
    """
    return {
        'degree': np.array([dc for dc in nx.degree_centrality(graph).values()]),
        'closeness': np.array([dc for dc in nx.closeness_centrality(graph).values()]),
        'betweenness': np.array([dc for dc in nx.betweenness_centrality(graph).values()])
    }


def names_by_cluster(G, clusters, n_clusters):
    """ 
    returns surnames of people grouped by clusters
    used to interpret quality of clustering 
    """
    names = {cluster: [] for cluster in range(n_clusters)}

    for n, cluster in zip(G.nodes, clusters):
        names[cluster].append(G.nodes[n]['last_name'])

    return names


def to_pandas(graph, centralities) -> pd.DataFrame:
    """ creates a dataframe to filter data more efficiently """
    names = []
    cities = []
    genders = []
    relations = []

    for idx in graph.nodes:
        names.append(graph.nodes[idx]['first_name'] +
                     " "+graph.nodes[idx]['last_name'])
        genders.append(gender_chart[graph.nodes[idx]['gender']])
        cities.append(graph.nodes[idx]['city'])
        cur_relation = graph.nodes[idx]['relation']
        relations.append(relation_chart.get(cur_relation, None))

    d = {'node_id': graph.nodes,
         'name': names,
         'gender': genders,
         'relation': relations,
         'city': cities,
         'deg': centralities['degree'],
         'closeness': centralities['closeness'],
         'betweenness': centralities['betweenness']
         }

    return pd.DataFrame(data=d)


def norm_laplacian(A):
    """ Computes normalized Laplacian  """
    D = np.diag(A.sum(axis=1))
    L = D - A
    hat_D = fractional_matrix_power(D, -0.5)
    N = np.matmul(np.matmul(hat_D, L), hat_D)
    return N


def spectral_embedding(L, n_components):
    """
    forms the Laplacian eigenmaps, use the K smallest eigenvectors
    (excluding the smallest), where K = n_components

    returns:
    a matrix where columns are eigenvectors, rows are node embeddings
    """
    w, v = np.linalg.eigh(L)
    return v[np.argsort(w)][:, 1:1+n_components]


def spectral_clustering(G, n_clusters, n_components):
    """ 
    clusters using normalized laplacian and Kmeans

    G: graph
    n_cluster: number of clusters to receive
    n_components: size (second dimension) of embeddings

    returns node2cluster, 
    can be used with zip(G.nodes, node2cluster)
    """
    A = nx.to_numpy_array(G)
    L = norm_laplacian(A)
    embedding = spectral_embedding(L, n_components)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embedding)
    return kmeans.labels_


def edge_betw_modularity(G, n):
    """
    calculates modularity for each split to define
    the most suitable number of clusters
    """
    com_gen = nx.algorithms.community.girvan_newman(G)
    mods = np.zeros(n, dtype='float64')
    for i in tqdm(range(n)):
        communities = next(com_gen)
        mods[i] = nx.algorithms.community.modularity(
            G, communities)

    return np.array(mods)


def sim_matrices(graph: nx.Graph):
    """
    returns similarity matrices for the given graph
    A: adjacency matrix
    corr: Pearson correlation
    J: Jaccard similarity
    cos: Cosine similarity
    """
    A = nx.to_numpy_array(graph)
    corr = np.corrcoef(A)
    J = np.zeros(A.shape)
    for i, j, c in nx.jaccard_coefficient(nx.from_numpy_array(A)):
        J[i, j] = c
        J[j, i] = c
    cos = cosine_similarity(A)
    return A, corr, J, cos


def empirical_cdf(g: nx.Graph) -> np.array:
    """ calculates probabilities for degrees """
    nums = nx.degree_histogram(g)
    probs = np.empty(len(nums))
    tmp_total = 0
    for i in range(len(nums)):
        tmp_total += nums[i]
        probs[i] = tmp_total / g.number_of_nodes()
    return probs


gender_chart = {
    1: "female",
    2: "male",
    0: "NS"
}

relation_chart = {
    1: "single",
    2: "in a relationship",
    3: "engaged",
    4: "married",
    5: "complicated",
    6: "actively searching",
}
