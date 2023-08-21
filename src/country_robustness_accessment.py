import networkx as nx
import numpy as np
import pandas as pd


def country_structural_robustness_scenario1():
    df = pd.read_csv('../output/port_centrality.csv')
    country = list(set(df['ISO3'].values.tolist()))
    df2 = pd.read_csv('../data/edges of a synthetic network.csv')
    G = nx.Graph()
    for i in range(0, df2.shape[0]):
        G.add_edge(df2.loc[i]['port1'], df2.loc[i]['port2'])
    R = []
    for c in country:
        country_nodes = df.loc[df['ISO3'] == c]
        country_nodes = country_nodes['ports'].values.tolist()
        C_LCC = pd.DataFrame([], columns=['id'])
        seed = 0
        for k in range(0, 1000):
            G1 = G.copy()
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
                G1.remove_node(degree.loc[i]['ports'])
                largest = list(max(nx.connected_components(G1), key=len))
                nodes_in_LCC = list(set(largest).intersection(set(country_nodes)))
                other_nodes_in_LCC = list(set(largest).difference(set(country_nodes)))
                if (len(nodes_in_LCC) != 0) and (len(other_nodes_in_LCC) != 0):
                    m = [i + 1, len(nodes_in_LCC)]
                    LCC.append(m)
                if (len(nodes_in_LCC) == 0) or (len(other_nodes_in_LCC) == 0):
                    m = [i + 1, 0]
                    LCC.append(m)
            LCC1 = pd.DataFrame(LCC, columns=['id', 'LCC-' + str(k)])
            print('country:',c, 'times:',str(k), 'robustness:',np.trapz(LCC1['LCC-' + str(k)], LCC1['id'] / 907))  # 针对degree移除节点，重复30次
            C_LCC = pd.merge(C_LCC, LCC1, left_on=['id'], right_on=['id'], how='outer')
        C_LCC['mean'] = C_LCC.iloc[:, 1:].mean(axis=1)
        r = [c, np.trapz(C_LCC['mean'], C_LCC['id'] / 907)]
        R.append(r)
        print(r)
        #C_LCC.to_csv('../output/' + str(c) + '_robustness_scenario1.csv')
    R = pd.DataFrame(R, columns=['ISO3', 'R1'])
    R.to_csv('../output/countries_robustness_scenario1.csv')


def country_structural_robustness_scenario2():
    df = pd.read_csv('../output/port_centrality.csv')
    country = list(set(df['ISO3'].values.tolist()))
    df2 = pd.read_csv('../data/edges of a synthetic network.csv')
    G = nx.Graph()
    for i in range(0, df2.shape[0]):
        G.add_edge(df2.loc[i]['port1'], df2.loc[i]['port2'])
    R = []
    for c in country:
        df_c = df.loc[df['ISO3'] != c]
        country_nodes = df.loc[df['ISO3'] == c]
        country_nodes = country_nodes['ports'].values.tolist()
        C_LCC = pd.DataFrame([], columns=['id'])
        seed = 0
        for k in range(0, 1000):
            G1 = G.copy()
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
                G1.remove_node(degree.loc[i]['ports'])
                largest = list(max(nx.connected_components(G1), key=len))
                nodes_in_LCC = list(set(largest).intersection(set(country_nodes)))
                other_nodes_in_LCC = list(set(largest).difference(set(country_nodes)))
                if (len(nodes_in_LCC) != 0) and (len(other_nodes_in_LCC) != 0):
                    m = [i + 1, len(nodes_in_LCC)]
                    LCC.append(m)
                if (len(nodes_in_LCC) == 0) or (len(other_nodes_in_LCC) == 0):
                    m = [i + 1, 0]
                    LCC.append(m)
            LCC1 = pd.DataFrame(LCC, columns=['id', 'LCC-' + str(k)])
            C_LCC = pd.merge(C_LCC, LCC1, left_on=['id'], right_on=['id'], how='outer')
            print('country:', c, 'times:', str(k), 'robustness:',
                  np.trapz(LCC1['LCC-' + str(k)], LCC1['id'] / 907))

        C_LCC['mean'] = C_LCC.iloc[:, 1:].mean(axis=1)
        r = [c, np.trapz(C_LCC['mean'], C_LCC['id'] / (907 - len(country_nodes)))]
        R.append(r)
        #C_LCC.to_csv('../output/' + str(c) + '_robustness_scenario2.csv')
        R = pd.DataFrame(R, columns=['ISO3', 'R2'])
        R.to_csv('../output/countries_robustness_scenario2.csv')


def main():
    country_structural_robustness_scenario1()
    country_structural_robustness_scenario2()


if __name__ == '__main__':
    main()
