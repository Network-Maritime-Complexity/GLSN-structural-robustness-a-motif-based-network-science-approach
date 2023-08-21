import networkx as nx
import numpy as np
import pandas as pd
import random
from network_robustness_accessment import delibrate_attack


def motif_international_unconnect_edge():
    df1 = pd.read_csv('../data/edges of a synthetic network.csv').astype(str)
    node_ISO3 = pd.read_csv('../output/port_centrality.csv').astype(str)
    node_ISO3 = node_ISO3[['ports', 'ISO3', 'degree']]
    G = nx.Graph()
    for i in range(0, df1.shape[0]):
        G.add_edge(df1.loc[i]['port1'], df1.loc[i]['port2'])
    edge = pd.DataFrame(G.edges(), columns=['port1', 'port2'])
    edge1 = edge[['port2', 'port1']]
    edge1.columns = ['port1', 'port2']
    edge = pd.merge(edge, edge1, left_on=['port1', 'port2'], right_on=['port1', 'port2'], how='outer')
    result = pd.DataFrame(nx.all_pairs_shortest_path_length(G), columns=['port1', 'dict'])
    result = result.drop('dict', 1).assign(**pd.DataFrame(result.dict.values.tolist()))
    result = result.set_index('port1')
    result = result.stack()
    result = result.reset_index().rename(columns={'level_1': 'port2', 0: 'stp'})
    result = result.loc[(result['stp'] == 3) | (result['stp'] == 2)]
    result.loc[result['port1'] >= result['port2'], 'edge'] = result['port1'] + '--' + result['port2']
    result.loc[result['port1'] < result['port2'], 'edge'] = result['port2'] + '--' + result['port1']
    result.drop_duplicates(subset=['edge'], keep='first', inplace=True)
    result = pd.concat([result, edge, edge]).drop_duplicates(subset=['port1', 'port2'], keep=False)
    result = pd.merge(result, node_ISO3, left_on=['port1'], right_on=['ports'], how='left')
    result = pd.merge(result, node_ISO3, left_on=['port2'], right_on=['ports'], how='left')
    result = result.loc[result['ISO3_x'] != result['ISO3_y']]
    print(result.shape[0])
    result['sum'] = result['degree_x'] + result['degree_y']
    result = result[['port1', 'port2', 'edge', 'ISO3_x', 'ISO3_y', 'sum']]
    result.columns = ['port1', 'port2', 'edge', 'ISO3_1', 'ISO3_2', 'sum']
    result.to_csv('../output/motif_international_unconnect_edge.csv')


def MLL(num, df, edge):
    df['p'] = 1 / df['degree']
    nodes = df['ports'].values.tolist()
    df['p'] = df['p'] / df['p'].sum()
    p = df['p'].values.tolist()
    dis_edge = edge['edge'].values.tolist()
    EDGE = []
    i = 0
    while len(EDGE) < num:
        port1 = np.random.choice(a=nodes, p=p, size=1)
        port2 = np.random.choice(a=nodes, p=p, size=1)
        port1 = port1[0]
        port2 = port2[0]
        if port1 >= port2:
            e = str(port1) + '--' + str(port2)
        if port1 < port2:
            e = str(port2) + '--' + str(port1)
        if e in dis_edge:
            dis_edge.remove(e)
            m = [port1, port2]
            EDGE.append(m)
    EDGE1 = pd.DataFrame(EDGE, columns=['port1', 'port2'])
    return EDGE1


def MLDF(num, edge):
    NUM = list(set(edge['sum']))
    NUM.sort()
    EDGE = []
    i = 0
    while len(EDGE) < num:
        motif = list(edge.loc[edge['sum'] == NUM[i]].index)
        i = i + 1
        while len(EDGE) < num and len(motif) > 0:
            m = random.sample(motif, 1)
            motif.remove(m[0])
            a = [edge.loc[m[0]]['port1'], edge.loc[m[0]]['port2'], edge.loc[m[0]]['edge']]
            EDGE.append(a)
    EDGE = pd.DataFrame(EDGE, columns=['port1', 'port2', 'edge'])
    EDGE = EDGE.astype(str)
    return EDGE


def MHH(num, df, edge):
    df['p'] = df['degree']
    nodes = df['ports'].values.tolist()
    df['p'] = df['p'] / df['p'].sum()
    p = df['p'].values.tolist()
    dis_edge = edge['edge'].values.tolist()
    EDGE = []
    while len(EDGE) < num:
        port1 = np.random.choice(a=nodes, p=p, size=1)
        port2 = np.random.choice(a=nodes, p=p, size=1)
        port1 = port1[0]
        port2 = port2[0]
        if port1 >= port2:
            e = str(port1) + '--' + str(port2)
        if port1 < port2:
            e = str(port2) + '--' + str(port1)
        if e in dis_edge:
            dis_edge.remove(e)
            m = [port1, port2]
            EDGE.append(m)
    EDGE1 = pd.DataFrame(EDGE, columns=['port1', 'port2'])
    return EDGE1


def MHDF(num, edge):
    NUM = list(set(edge['sum']))
    NUM.sort(reverse=True)
    EDGE = []
    i = 0
    while len(EDGE) < num:
        motif = list(edge.loc[edge['sum'] == NUM[i]].index)
        i = i + 1
        while len(EDGE) < num and len(motif) > 0:
            m = random.sample(motif, 1)
            motif.remove(m[0])
            a = [edge.loc[m[0]]['port1'], edge.loc[m[0]]['port2'], edge.loc[m[0]]['edge']]
            EDGE.append(a)
    EDGE = pd.DataFrame(EDGE, columns=['port1', 'port2', 'edge'])
    return EDGE


def Random(num, edge):
    EDGE = edge.sample(num)
    return EDGE


def net_robust_improve():
    num = [50, 100, 150, 200, 250, 300, 350,400]
    un_edge = pd.read_csv('../output/motif_international_unconnect_edge.csv')
    node = pd.read_csv('../output/port_centrality.csv')
    edge = pd.read_csv('../data/edges of a synthetic network.csv')
    G = nx.Graph()
    for i in edge.index:
        G.add_edge(edge.loc[i]['port1'], edge.loc[i]['port2'])
    R_rate = []
    for n in num:
        R_MLL = 0
        R_MLDF = 0
        R_MHH = 0
        R_MHDF = 0
        R_Random = 0
        for i in range(0, 100):
            G0 = G.copy()
            edge_MLL = MLL(n, node, un_edge)
            for j in edge_MLL.index:
                G0.add_edge(edge_MLL.loc[j]['port1'], edge_MLL.loc[j]['port2'])
            _, r_mll = delibrate_attack('degree', node, G0, 10)
            R_MLL = R_MLL+(r_mll-267.02)/267.02
            G1 = G.copy()
            edge_MHH = MHH(n, node, un_edge)
            for j in edge_MHH.index:
                G1.add_edge(edge_MHH.loc[j]['port1'], edge_MHH.loc[j]['port2'])
            _, r_mhh = delibrate_attack('degree', node, G1, 10)
            R_MHH = R_MHH+(r_mhh-267.02)/267.02
            G2 = G.copy()
            edge_MLDF = MLDF(n, un_edge)
            for j in edge_MLDF.index:
                G2.add_edge(edge_MLDF.loc[j]['port1'], edge_MLDF.loc[j]['port2'])
            _, r_mldf = delibrate_attack('degree', node, G2, 10)
            R_MLDF = R_MLDF + (r_mldf-267.02)/267.02
            G3 = G.copy()
            edge_MHDF = MHDF(n, un_edge)
            for j in edge_MHDF.index:
                G3.add_edge(edge_MHDF.loc[j]['port1'], edge_MHDF.loc[j]['port2'])
            _, r_mhdf = delibrate_attack('degree', node, G3, 10)
            R_MHDF = R_MHDF + (r_mhdf-267.02)/267.02
            G4 = G.copy()
            edge_random = Random(n, un_edge)
            for j in edge_random.index:
                G4.add_edge(edge_random.loc[j]['port1'], edge_random.loc[j]['port2'])
            _, r_random = delibrate_attack('degree', node, G4, 10)
            R_Random = R_Random + (r_random-267.02)/267.02
        r = [n, R_MLL, R_MLDF, R_MLL, R_MHDF, R_Random]
        R_rate.append(r)
    R_rate = pd.DataFrame(R_rate, columns = ['number_of_edges', 'R_rate_MLL', 'R_rate_MLDF', 'R_rate_MHH', 'R_rate_MHDF', 'R_rate_Random'])
    R_rate = R_rate[['number_of_edges', 'R_rate_MLL', 'R_rate_MLDF', 'R_rate_MHH', 'R_rate_MHDF', 'R_rate_Random']]/100
    R_rate.to_csv('../output/glsn_robustness_improvement_rate.csv')


def main():
    motif_international_unconnect_edge()
    net_robust_improve()


