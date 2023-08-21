import networkx as nx
import numpy as np
import pandas as pd
import port_centrality
import network_robustness_accessment
import network_robustness_improvement
import country_robustness_accessment
import country_robustness_improvement


if __name__ == '__main__':
    # #---------- results-------# #
    port_centrality.main()
    network_robustness_accessment.main()
    network_robustness_improvement.main()
    country_robustness_accessment.main()
    country_robustness_improvement.main()
