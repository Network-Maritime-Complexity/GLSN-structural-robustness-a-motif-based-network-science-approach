import networkx as nx
import numpy as np
import pandas as pd


def add_edge(df, n, country_node, dis_edge):
    nodes1 = df['ports'].values.tolist()
    p1 = df['p'].values.tolist()
    nodes2 = country_node['ports'].values.tolist()
    p2 = country_node['p'].values.tolist()
    EDGE = []
    while len(EDGE) < n:
        node1 = np.random.choice(a=nodes1, p=p1, size=1)
        node2 = np.random.choice(a=nodes2, p=p2, size=1)
        node1 = node1[0]
        node2 = node2[0]
        if node1 >= node2:
            e = str(node1) + '--' + str(node2)
        if node1 < node2:
            e = str(node2) + '--' + str(node1)
        if e in dis_edge:
            dis_edge.remove(e)
            m = [node1, node2]
            EDGE.append(m)
    EDGE1 = pd.DataFrame(EDGE, columns=['node1', 'node2'])
    return EDGE1


def MLL_scenario1():
    df = pd.read_csv('../output/port_centrality.csv')
    df1 = pd.read_csv('../output/motif_international_unconnect_edge.csv')
    country = list(set(df['ISO3'].values.tolist()))
    df2 = pd.read_csv('../data/edges of a synthetic network.csv')
    G = nx.Graph()
    for i in range(0, df2.shape[0]):
        G.add_edge(df2.loc[i]['port1'], df2.loc[i]['port2'])
    R1_MLL = []
    for c in country:
        c1 = df.loc[df['ISO3'] == c]
        c1['p'] = 1 / c1['degree']
        c1['p'] = c1['p'] / c1['p'].sum()
        df_c = df.loc[df['ISO3'] != c]
        df_c['p'] = 1 / df_c['degree']
        df_c['p'] = df_c['p'] / df_c['p'].sum()
        country_nodes = c1['ports'].values.tolist()
        C_LCC = pd.DataFrame([], columns=['id'])
        for n in range(0, 100):
            c_edge = df1.loc[((df1['ISO3_1'] == c) & (df1['ISO3_2'] != c)) | (
                        (df1['ISO3_1'] != c) & (df1['ISO3_2'] == c))]
            dis_edge = c_edge['edge'].values.tolist()
            if c_edge.shape[0] >= 10:
                new_edge = add_edge(df_c, 10, c1, dis_edge)
                G1 = G.copy()
                for index in new_edge.index:
                    G1.add_edge(new_edge.loc[index]['node1'], new_edge.loc[index]['node2'])
                seed = 0
                for k in range(0, 10):
                    print('country:',c, ' add_edge_time:', n,' remove_time:', k)
                    G2 = G1.copy()
                    degree = df.copy()
                    degree = degree.sample(frac=1, random_state=seed)
                    degree.reset_index(inplace=True)
                    seed = seed + 1
                    degree.sort_values(by='degree', ascending=False, inplace=True)
                    degree.reset_index(inplace=True)
                    LCC = []
                    m = [0, len(country_nodes)]
                    LCC.append(m)
                    for i in range(0, degree.shape[0] - 1):
                        G2.remove_node(degree.loc[i]['ports'])
                        largest = list(max(nx.connected_components(G2), key=len))
                        nodes_in_LCC = list(set(largest).intersection(set(country_nodes)))
                        other_nodes_in_LCC = list(set(largest).difference(set(country_nodes)))
                        if (len(nodes_in_LCC) != 0) and (len(other_nodes_in_LCC) != 0):
                            m = [i + 1, len(nodes_in_LCC)]
                            LCC.append(m)
                        if (len(nodes_in_LCC) == 0) or (len(other_nodes_in_LCC) == 0):
                            m = [i + 1, 0]
                            LCC.append(m)
                    LCC1 = pd.DataFrame(LCC, columns=['id', 'LCC' + str(n) + '-' + str(k)])
                    C_LCC = pd.merge(C_LCC, LCC1, left_on=['id'], right_on=['id'], how='outer')
        C_LCC['mean'] = C_LCC.iloc[:, 1:].mean(axis=1)
        r = [c, np.trapz(C_LCC['mean'], C_LCC['id'] / 907)]
        print(r)
        #C_LCC.to_csv('../results/' + str(c) + '_robustness_improvement_1.csv')
        R1_MLL.append(r)
    R1_MLL = pd.DataFrame(R1_MLL, columns=['ISO3', 'R1_MLL'])
    return R1_MLL


def MLL_scenario2():
    df = pd.read_csv('../output/port_centrality.csv')
    df1 = pd.read_csv('../output/motif_international_unconnect_edge.csv')
    country = list(set(df['ISO3'].values.tolist()))
    df2 = pd.read_csv('../data/edges of a synthetic network.csv')
    G = nx.Graph()
    for i in range(0, df2.shape[0]):
        G.add_edge(df2.loc[i]['port1'], df2.loc[i]['port2'])
    R2_MLL = []
    for c in country:
        c1 = df.loc[df['ISO3'] == c]
        c1['p'] = 1 / c1['degree']
        c1['p'] = c1['p'] / c1['p'].sum()
        df_c = df.loc[df['ISO3'] != c]
        df_c['p'] = 1 / df_c['degree']
        df_c['p'] = df_c['p'] / df_c['p'].sum()
        country_nodes = c1['ports'].values.tolist()
        C_LCC = pd.DataFrame([], columns=['id'])
        for n in range(0, 100):
            c_edge = df1.loc[((df1['ISO3_1'] == c) & (df1['ISO3_2'] != c)) | (
                        (df1['ISO3_1'] != c) & (df1['ISO3_2'] == c))]
            dis_edge = c_edge['edge'].values.tolist()
            if c_edge.shape[0] >= 10:
                new_edge = add_edge(df_c, 10, c1, dis_edge)
                G1 = G.copy()
                for index in new_edge.index:
                    G1.add_edge(new_edge.loc[index]['node1'], new_edge.loc[index]['node2'])
                seed = 0
                for k in range(0, 10):
                    print('country:',c, ' add_edge_time:', n,' remove_time:', k)
                    G2 = G1.copy()
                    degree = df_c.copy()
                    degree = degree.sample(frac=1, random_state=seed)
                    degree.reset_index(inplace=True)
                    seed = seed + 1
                    degree.sort_values(by='degree', ascending=False, inplace=True)
                    degree.reset_index(inplace=True)
                    LCC = []
                    m = [0, len(country_nodes)]
                    LCC.append(m)
                    for i in range(0, degree.shape[0]):
                        G2.remove_node(degree.loc[i]['ports'])
                        largest = list(max(nx.connected_components(G2), key=len))
                        nodes_in_LCC = list(set(largest).intersection(set(country_nodes)))
                        other_nodes_in_LCC = list(set(largest).difference(set(country_nodes)))
                        if (len(nodes_in_LCC) != 0) and (len(other_nodes_in_LCC) != 0):
                            m = [i + 1, len(nodes_in_LCC)]
                            LCC.append(m)
                        if (len(nodes_in_LCC) == 0) or (len(other_nodes_in_LCC) == 0):
                            m = [i + 1, 0]
                            LCC.append(m)
                    LCC1 = pd.DataFrame(LCC, columns=['id', 'LCC' + str(n) + '-' + str(k)])
                    C_LCC = pd.merge(C_LCC, LCC1, left_on=['id'], right_on=['id'], how='outer')
        C_LCC['mean'] = C_LCC.iloc[:, 1:].mean(axis=1)
        r = [c, np.trapz(C_LCC['mean'], C_LCC['id'] / 907)]
        print(r)
        # C_LCC.to_csv('../results/' + str(c) + '_robustness_improvement_1.csv')
        R2_MLL.append(r)
    R2_MLL = pd.DataFrame(R2_MLL, columns=['ISO3', 'R2_MLL'])
    return R2_MLL


def country_robustness_improve_rate():
    R1_MLL = MLL_scenario1()
    R2_MLL = MLL_scenario2()
    mll = pd.merge(R1_MLL, R2_MLL, left_on=['ISO3'], right_on=['ISO3'], how='left')
    R1 = pd.read_csv('../output/countries_robustness_scenario1.csv')
    R2 = pd.read_csv('../output/countries_robustness_scenario2.csv')
    df1 = pd.merge(mll, R1, left_on=['ISO3'], right_on=['ISO3'], how='left')
    df1 = pd.merge(df1, R2, left_on=['ISO3'], right_on=['ISO3'], how='left')
    df1['R1_improve_rate'] = (df1['R1_MLL'] - df1['R1'])/df1['R1']
    df1['R2_improve_rate'] = (df1['R2_MLL'] - df1['R2'])/df1['R2']
    return df1


def country_info(df1):
    df2 = pd.read_csv('../data/edges of a synthetic network.csv')
    df3 = pd.read_csv('../data/nodes of a synthetic network.csv')
    df31 = df3[['ports', 'ISO3']]
    df2 = pd.merge(df2, df31, left_on=['port1'], right_on=['ports'], how='left')
    df2 = pd.merge(df2, df31, left_on=['port2'], right_on=['ports'], how='left')
    for i in range(0, df3.shape[0]):
        a = df3.loc[i]['ports']
        b = df3.loc[i]['ISO3']
        degree1 = df2.loc[(df2['port1'] == a) & (df2['ISO3_y'] != b)]
        node1 = degree1['port2'].values.tolist()
        degree2 = df2.loc[(df2['port2'] == a) & (df2['ISO3_x'] != b)]
        node2 = degree2['port1'].values.tolist()
        nodes = list(set(node1).union(set(node2)))
        df3.loc[i, 'degree_nation'] = len(nodes)
    for i in range(0, df1.shape[0]):
        a = df3.loc[df3['ISO3'] == df1.loc[i]['ISO3']]
        df1.loc[i, 'number of domestic ports'] = a.shape[0]
        df1.loc[i, 'number of international links'] = a['degree_nation'].sum()
    df1.to_csv('../output/countries_robustness_improve_rate.csv')


def main():
    df1 = country_robustness_improve_rate()
    country_info(df1)






