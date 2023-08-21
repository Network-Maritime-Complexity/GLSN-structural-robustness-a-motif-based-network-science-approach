import networkx as nx
import numpy as np
import pandas as pd
import random
from copy import deepcopy


def delibrate_attack(feature, df1, G, n):
    LCC1 = pd.DataFrame([], columns=['id'])
    seed = 0
    for j in range(0, n):
        G2 = G.copy()
        degree = df1.copy()
        degree = degree.sample(frac=1, random_state=seed)
        degree.reset_index(inplace=True)
        seed = seed + 1
        degree.sort_values(by=str(feature), ascending=False, inplace=True)
        degree.reset_index(inplace=True)
        LCC = []
        m = [0, 907]
        LCC.append(m)
        for i in range(0, degree.shape[0] - 1):
            G2.remove_node(degree.loc[i]['ports'])
            largest = len(max(nx.connected_components(G2), key=len))
            m = [i + 1, largest]
            LCC.append(m)
        LCC = pd.DataFrame(LCC, columns=['id', 'LCC' + str(j)])
        LCC1 = pd.merge(LCC1, LCC, left_on=['id'], right_on=['id'], how='outer')
    LCC1['LCC-'+str(feature)] = LCC1.iloc[:, 1:].mean(axis=1)
    LCC1 = LCC1[['id', 'LCC-'+str(feature)]]
    return LCC1, np.trapz(LCC1['LCC-'+str(feature)], LCC1['id']/907)


def random_attack(G, df1, n):
    LCC1 = pd.DataFrame([], columns=['id'])
    seed = 0
    for j in range(0, n):
        G2 = G.copy()
        degree = df1.copy()
        degree = degree.sample(frac=1, random_state=seed)
        degree.reset_index(inplace=True)
        seed = seed + 1
        degree.reset_index(inplace=True)
        LCC = []
        m = [0, 907]
        LCC.append(m)
        for i in range(0, degree.shape[0] - 1):
            G2.remove_node(degree.loc[i]['ports'])
            largest = len(max(nx.connected_components(G2), key=len))
            m = [i + 1, largest]
            LCC.append(m)
        LCC = pd.DataFrame(LCC, columns=['id', 'LCC' + str(j)])
        LCC1 = pd.merge(LCC1, LCC, left_on=['id'], right_on=['id'], how='outer')
    LCC1['LCC-Random'] = LCC1.iloc[:, 1:].mean(axis=1)
    LCC1 = LCC1[['id', 'LCC-Random']]
    return LCC1, np.trapz(LCC1['LCC-Random'], LCC1['id'] / 907)


def keep_degree(G, df, n):
    P_net = G
    rank_data = df
    rank_id = list(rank_data['ports'])
    id_num = list(rank_data['MC'])
    num_group_dict = dict()
    for index_i in range(df.shape[0]):
        node_num = id_num[index_i]
        num_group_dict.setdefault(node_num, []).append(rank_id[index_i])
    num_group_dict = dict(sorted(num_group_dict.items(), key=lambda d: d[0], reverse=True))
    rank_id = []
    for k in num_group_dict.keys():
        rank_id.extend(num_group_dict[k])
    degree_group_dict = dict()
    for node in P_net.nodes:
        node_degree = nx.degree(P_net, node)
        degree_group_dict.setdefault(node_degree, []).append(node)
    save_data_all = pd.DataFrame()
    # save_data['del_num'] = list(range(977))
    for my_seed in range(n):
        all_max_connected = []
        keep_degree_node = []
        deg_dict = deepcopy(degree_group_dict)
        for node_id in rank_id:
            temp_degree = nx.degree(P_net, node_id)
            random.seed(my_seed)
            temp_node = node_id
            temp_list = []
            temp_list.extend(deg_dict[temp_degree])
            if node_id in temp_list:
                temp_list.remove(node_id)
            if len(temp_list) != 0:
                temp_node = random.choice(temp_list)
            keep_degree_node.append(temp_node)
            deg_dict[temp_degree].remove(temp_node)
        for del_num in range(df.shape[0]):
            sub_max_bcc = P_net.copy()
            sub_max_bcc.remove_nodes_from(keep_degree_node[0:del_num])
            largest_cc = max(nx.connected_components(sub_max_bcc), key=len)
            all_max_connected.append(len(largest_cc))
        save_data_all['largest_cc{0}'.format(my_seed)] = all_max_connected
    save_data = pd.DataFrame()
    save_data['LCC-keep-degree'] = save_data_all.mean(1)
    save_data = save_data['LCC-keep-degree']
    save_data.reset_index(inplace=True)
    save_data.columns = ['id', 'LCC-keep-degree']
    return save_data, save_data['LCC-keep-degree'].sum()/907


def network_robustness():
    feature = ['MC', 'degree', 'eigenvector']
    df = pd.read_csv('../output/port_centrality.csv')
    edge = pd.read_csv('../data/edges of a synthetic network.csv')
    LCC_all = pd.DataFrame([], columns=['id'])
    G = nx.Graph()
    for i in range(0, edge.shape[0]):
        G.add_edge(edge.loc[i]['port1'], edge.loc[i]['port2'])
    LCC, robustness = keep_degree(G, df, 1000)
    LCC_all = pd.merge(LCC_all, LCC, left_on=['id'], right_on=['id'], how='outer')
    print('feature:keep_degree', ' network robustness:', robustness)
    LCC, robustness = random_attack(G, df, 1000)
    LCC_all = pd.merge(LCC_all, LCC, left_on=['id'], right_on=['id'], how='outer')
    print('feature:random', ' network robustness:', robustness)
    for x in feature:
        LCC, robustness = delibrate_attack(x, df, G, 1000)
        print('feature:', str(x), ' network robustness:', robustness)
        LCC_all = pd.merge(LCC_all, LCC, left_on=['id'], right_on=['id'], how='outer')
    LCC_all.to_csv('../output/glsn_robustness.csv')


def main():
    network_robustness()
