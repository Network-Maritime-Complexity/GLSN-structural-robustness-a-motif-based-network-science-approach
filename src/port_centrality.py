import random
import pandas as pd
import networkx as nx
import numpy as np
from itertools import combinations
from itertools import product


def port_basic_centrality():
    df1 = pd.read_csv('../data/edges of a synthetic network.csv')
    df2 = pd.read_csv('../data/nodes of a synthetic network.csv').astype(str)
    df1 = df1.astype(str)
    G = nx.Graph()
    for i in range(0, df1.shape[0]):
        G.add_edge(df1.loc[i]['port1'], df1.loc[i]['port2'])
    D = pd.DataFrame(nx.degree(G), columns=['ports', 'degree'])
    B = pd.DataFrame(pd.Series(nx.betweenness_centrality(G)))
    B.reset_index(inplace=True)
    B.columns = ['ports', 'betweenness']
    E = pd.DataFrame(pd.Series(nx.eigenvector_centrality(G)))
    E.reset_index(inplace=True)
    E.columns = ['ports', 'eigenvector']
    PC = pd.merge(D, B, left_on=['ports'], right_on=['ports'],how='left')
    PC = pd.merge(PC, E, left_on=['ports'], right_on=['ports'], how='left')
    PC = pd.merge(PC, df2, left_on=['ports'], right_on=['ports'], how='left')
    return PC


def find_T1(G):
    nodes = list(G.nodes)
    edge = pd.DataFrame(G.edges, columns=['node1', 'node2'])
    edge_1 = edge[['node2', 'node1']]
    edge_1.columns = ['node1', 'node2']
    edge = pd.merge(edge_1, edge, left_on=['node1', 'node2'], right_on=['node1', 'node2'], how='outer')
    T1 = pd.DataFrame([], columns=['node1', 'node2', 'node3'])
    for x in nodes:
        A = G.neighbors(x)
        B = pd.DataFrame(combinations(A, 2), columns=['node1', 'node2'])
        C = pd.concat([B, edge, edge]).drop_duplicates(subset=['node1', 'node2'], keep=False)
        C['node3'] = x
        T1 = pd.concat([T1, C])
    return T1


def find_M4(G, T1):
    M4 = pd.DataFrame([], columns=['node1', 'node2', 'node3', 'node4', 'motif', 'num'])
    for i in G.nodes:
        M = T1.loc[T1['node3'] == i]
        if M.shape[0] != 0:
            M['nei1'] = M.apply(lambda x: list(set(G.neighbors(x['node1'])).intersection(set(G.neighbors(x['node2'])))),
                                axis=1)
            M['nei2'] = M['node3'].apply(lambda x: list(G.neighbors(x)))
            M['node4'] = M.apply(lambda x: list(set(x['nei1']).difference(set(x['nei2']))), axis=1)
            M = M[['node1', 'node2', 'node3', 'node4']]
            M = M[['node1', 'node2', 'node3', 'node4']].explode('node4')
            M['motif'] = M[['node1', 'node2', 'node3', 'node4']].apply(lambda x: sorted(list(set(x.values.tolist()))),
                                                                       axis=1)
            M['num'] = M['motif'].apply(lambda x: len(x))
            M = M.loc[M['num'] == 4]
            M['motif'] = M['motif'].astype(str)
            M4 = pd.concat([M4, M])
    M4.drop_duplicates(subset=['motif'], keep='first', inplace=True)
    print('M4', M4.shape[0])
    return M4


def find_M2(G, df):
    edge = pd.DataFrame(G.edges, columns=['node1', 'node2'])
    edge_1 = edge[['node2', 'node1']]
    edge_1.columns = ['node1', 'node2']
    edge = pd.merge(edge_1, edge, left_on=['node1', 'node2'], right_on=['node1', 'node2'], how='outer')
    M2 = pd.DataFrame([], columns=['node1', 'node2', 'node3', 'node4'])
    for i in range(0, df.shape[0]):
        a = df.loc[i]['port1']
        b = df.loc[i]['port2']
        nei1 = list(set(G.neighbors(a)).difference(set(G.neighbors(b))))
        nei2 = list(set(G.neighbors(b)).difference(set(G.neighbors(a))))
        c = pd.DataFrame(product(nei1, nei2), columns=['node1', 'node2'])
        C = pd.concat([c, edge, edge]).drop_duplicates(subset=['node1', 'node2'], keep=False)
        C.columns = ['node2', 'node4']
        C['node1'] = a
        C['node3'] = b
        C = C.astype(str)
        C['motif'] = C[['node1', 'node2', 'node3', 'node4']].apply(lambda x: sorted(list(set(x.values.tolist()))),
                                                                   axis=1)
        C['num'] = C['motif'].apply(lambda x: len(x))
        C = C.loc[C['num'] == 4]
        C['motif'] = C['motif'].astype(str)
        C.drop_duplicates(subset=['motif'], keep='first', inplace=True)
        M2 = pd.concat([M2, C])
    M2['motif'] = M2[['node1', 'node2', 'node3', 'node4']].apply(lambda x: sorted(x.values.tolist()), axis=1)
    M2['motif'] = M2['motif'].astype(str)
    M2.drop_duplicates(subset=['motif'], keep='first', inplace=True)
    print('M2', M2.shape[0])
    return M2


def find_M1_M3_M5(G):
    M1_1 = pd.DataFrame([], columns=['node1', 'node2', 'node3', 'node4'])
    M3_1 = pd.DataFrame([], columns=['node1', 'node2', 'node3', 'node4'])
    M5_1 = pd.DataFrame([], columns=['node1', 'node2', 'node3', 'node4'])
    for x in G.nodes():
        A = G.neighbors(x)
        B = pd.DataFrame(combinations(A, 3), columns=['n1', 'n2', 'n4'])
        B['num12'] = B[['n1', 'n2']].apply(
            lambda x: (G.subgraph(x.values.tolist())).number_of_edges(), axis=1)
        B['num14'] = B[['n1', 'n4']].apply(
            lambda x: (G.subgraph(x.values.tolist())).number_of_edges(), axis=1)
        B['num24'] = B[['n2', 'n4']].apply(
            lambda x: (G.subgraph(x.values.tolist())).number_of_edges(), axis=1)
        B['num'] = B[['num12', 'num14', 'num24']].sum(axis=1)
        B['n3'] = x
        M1 = B.loc[B['num'] == 0]
        M1 = M1[['n1', 'n2', 'n3', 'n4']]
        M1.columns = ['node1', 'node2', 'node3', 'node4']
        M3 = B.loc[B['num'] == 1]
        M3.loc[M3['num12'] == 1, 'node4'] = M3['n4']
        M3.loc[M3['num12'] == 1, 'node1'] = M3['n1']
        M3.loc[M3['num12'] == 1, 'node2'] = M3['n2']

        M3.loc[M3['num14'] == 1, 'node4'] = M3['n2']
        M3.loc[M3['num14'] == 1, 'node1'] = M3['n1']
        M3.loc[M3['num14'] == 1, 'node2'] = M3['n4']

        M3.loc[M3['num24'] == 1, 'node4'] = M3['n1']
        M3.loc[M3['num24'] == 1, 'node1'] = M3['n2']
        M3.loc[M3['num24'] == 1, 'node2'] = M3['n4']
        M3 = M3[['node1', 'node2', 'n3', 'node4']]
        M3.columns = ['node1', 'node2', 'node3', 'node4']

        M5 = B.loc[B['num'] == 2]
        M5.loc[M5['num12'] == 0, 'node2'] = M5['n4']
        M5.loc[M5['num12'] == 0, 'node1'] = M5['n1']
        M5.loc[M5['num12'] == 0, 'node4'] = M5['n2']

        M5.loc[M5['num14'] == 0, 'node2'] = M5['n2']
        M5.loc[M5['num14'] == 0, 'node1'] = M5['n1']
        M5.loc[M5['num14'] == 0, 'node4'] = M5['n4']

        M5.loc[M5['num24'] == 0, 'node2'] = M5['n1']
        M5.loc[M5['num24'] == 0, 'node1'] = M5['n4']
        M5.loc[M5['num24'] == 0, 'node4'] = M5['n2']
        M5 = M5[['node1', 'node2', 'n3', 'node4']]
        M5.columns = ['node1', 'node2', 'node3', 'node4']
        M1_1 = pd.concat([M1_1, M1])
        M3_1 = pd.concat([M3_1, M3])
        M5_1 = pd.concat([M5_1, M5])
    M5_1 = M5_1.astype(str)
    M5_1['motif'] = M5_1[['node1', 'node2', 'node3', 'node4']].apply(lambda x: sorted(x.values.tolist()), axis=1)
    M5_1['motif'] = M5_1['motif'].astype(str)

    M5_1.drop_duplicates(subset=['motif'], keep='first', inplace=True)
    print('M1', M1_1.shape[0])
    print('M3', M3_1.shape[0])
    print('M5', M5_1.shape[0])
    return M1_1, M3_1, M5_1

#def find_M6(G):
    #M6 = G.cliques(max=4, min=4)
    #M6 = pd.DataFrame(M6, columns=['node1', 'node2', 'node3', 'node4'])
    #return M6

def M1_centrality(M1, nodes):
    nodes_motif = []
    for node in nodes:
        num1 = M1.loc[(M1['node1'] == node)]
        num2 = M1.loc[(M1['node2'] == node)]
        num3 = M1.loc[(M1['node3'] == node)]
        num4 = M1.loc[(M1['node4'] == node)]
        a = [node, num1.shape[0], num2.shape[0], num3.shape[0], num4.shape[0], num3.shape[0]]
        nodes_motif.append(a)
    M1_centrality = pd.DataFrame(nodes_motif, columns=['nodes', 'node1_num', 'node2_num', 'node3_num', 'node4_num',
                                                     'M1_centrality'])
    return M1_centrality


def M2_centrality(M2, nodes):
    nodes_motif = []
    for node in nodes:
        num1 = M2.loc[(M2['node1'] == node)]
        num2 = M2.loc[(M2['node2'] == node)]
        num3 = M2.loc[(M2['node3'] == node)]
        num4 = M2.loc[(M2['node4'] == node)]
        a = [node, num1.shape[0], num2.shape[0], num3.shape[0], num4.shape[0], (num3.shape[0] + num1.shape[0]) * 2 / 3]
        nodes_motif.append(a)
    M2_centrality = pd.DataFrame(nodes_motif, columns=['nodes', 'node1_num', 'node2_num', 'node3_num', 'node4_num',
                                                     'M2_centrality'])
    return M2_centrality


def M3_centrality(M3, nodes):
    nodes_motif = []
    for node in nodes:
        num1 = M3.loc[(M3['node1'] == node)]
        num2 = M3.loc[(M3['node2'] == node)]
        num3 = M3.loc[(M3['node3'] == node)]
        num4 = M3.loc[(M3['node4'] == node)]
        a = [node, num1.shape[0], num2.shape[0], num3.shape[0], num4.shape[0], num3.shape[0] * 2 / 3]
        nodes_motif.append(a)
    M3_centrality = pd.DataFrame(nodes_motif, columns=['nodes', 'node1_num', 'node2_num', 'node3_num', 'node4_num',
                                                     'M3_centrality'])
    return M3_centrality


def M4_centrality(M4, nodes):
    nodes_motif = []
    for node in nodes:
        num1 = M4.loc[(M4['node1'] == node)]
        num2 = M4.loc[(M4['node2'] == node)]
        num3 = M4.loc[(M4['node3'] == node)]
        num4 = M4.loc[(M4['node4'] == node)]
        a = [node, num1.shape[0], num2.shape[0], num3.shape[0], num4.shape[0],
             (num2.shape[0] + num3.shape[0] + num1.shape[0] + num4.shape[0]) / 6]
        nodes_motif.append(a)
    M4_centrality = pd.DataFrame(nodes_motif, columns=['nodes', 'node1_num', 'node2_num', 'node3_num', 'node4_num',
                                                     'M4_centrality'])
    return M4_centrality


def M5_centrality(M5, nodes):
    nodes_motif = []
    for node in nodes:
        num1 = M5.loc[(M5['node1'] == node)]
        num2 = M5.loc[(M5['node2'] == node)]
        num3 = M5.loc[(M5['node3'] == node)]
        num4 = M5.loc[(M5['node4'] == node)]
        a = [node, num1.shape[0], num2.shape[0], num3.shape[0], num4.shape[0], (num2.shape[0] + num3.shape[0]) / 6]
        nodes_motif.append(a)
    M5_centrality = pd.DataFrame(nodes_motif, columns=['nodes', 'node1_num', 'node2_num', 'node3_num', 'node4_num',
                                                     'M5_centrality'])
    return M5_centrality


def MC():
    df1 = pd.read_excel('../data/edges of a synthetic network.xlsx')
    df1 = df1.astype(str)
    G = nx.Graph()
    for i in range(0, df1.shape[0]):
        G.add_edge(df1.loc[i]['port1'], df1.loc[i]['port2'])
    nodes = list(G.nodes())
    M2 = find_M2(G, df1)
    T1 = find_T1(G)
    M4 = find_M4(G, T1)
    M1, M3, M5 = find_M1_M3_M5(G)
    M1_C = M1_centrality(M1, nodes)
    M2_C = M2_centrality(M2, nodes)
    M3_C = M3_centrality(M3, nodes)
    M4_C = M4_centrality(M4, nodes)
    M5_C = M5_centrality(M5, nodes)
    M1_C = M1_C[['nodes', 'M1_centrality']]
    M2_C = M2_C[['nodes', 'M2_centrality']]
    M3_C = M3_C[['nodes', 'M3_centrality']]
    M4_C = M4_C[['nodes', 'M4_centrality']]
    M5_C = M5_C[['nodes', 'M5_centrality']]
    MC = pd.merge(M1_C, M2_C, left_on=['nodes'], right_on=['nodes'], how='left')
    MC = pd.merge(MC, M3_C, left_on=['nodes'], right_on=['nodes'], how='left')
    MC = pd.merge(MC, M4_C, left_on=['nodes'], right_on=['nodes'], how='left')
    MC = pd.merge(MC, M5_C, left_on=['nodes'], right_on=['nodes'], how='left')
    MC['MC'] = MC.iloc[:, 1:].sum(axis=1)
    MC = MC[['nodes', 'MC']]
    MC.columns = ['ports', 'MC']
    return MC


def main():
    Motif_centrality = MC()
    Pc = port_basic_centrality()
    port_centrality = pd.merge(Motif_centrality, Pc, left_on=['ports'], right_on=['ports'], how='left')
    port_centrality.to_csv('../output/port_centrality.csv')
