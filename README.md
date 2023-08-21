# **Assessing and improving the structural robustness of global liner shipping system: a motif-based network science approach**
This repository contains the code and data that support the findings of our article. Note: Raw data on world liner shipping services were provided by a third-party commercial database (Alphaliner, https://www.alphaliner.com/) and were used under the license for the current study, and so are not publicly available. However, all the data generated in our analysis are provided in the “results” folder of this repository.

Please cite our article if you use the code or data.

Mengqiao Xu, Wenhui Deng, Yifan Zhu, Linyuan LÜ. Assessing and improving the structural robustness of global liner shipping system: a motif-based network science approach. *Reliability Engineering & System Safety*. 2023
# **Overview**
The table below shows how the repository is structured. 

|Subdirectory|Description|
| - | - |
|data|This folder contains the data on a small synthetic network, for the convenience of testing our codes.|
|results|This folder contains the data generated for the main results of this paper.|
|src|<p>This folder contains all the scripts used for calculation, including:</p><p>the script "[port\_centrality.py](https://github.com/WenhuiDeng2000/GLSN-structural-robustness/blob/main/src/port_centrality.py)" for calculating the port motif centrality, port degree, port eigenvector centrality.</p><p>the script "[network\_robustness\_accessment.py](https://github.com/WenhuiDeng2000/GLSN-structural-robustness/blob/main/src/network_robustness_accessment.py)" for calculating the structural robustness of the GLSN under random attack strategy and under deliberate attack strategies based on degree centrality, eigenvector centrality, keep-degree, and MC.</p><p>the script "[country\_robustness\_accessment.py](https://github.com/WenhuiDeng2000/GLSN-structural-robustness/blob/main/src/country_robustness_accessment.py)" for calculating the structural robustness of each country under two degree-based deliberate attack scenarios.</p><p>the script "[network\_robustness\_improvement.py](https://github.com/WenhuiDeng2000/GLSN-structural-robustness/blob/main/src/network_robustness_improvement.py)" for calculating the improvement rate in GLSN structural robustness under five different strategies of link addition (i.e., random, MLDF, MHDF, MLL, and MHH), with the number of international links added ranging from 50 to 400</p><p>the script "[country\_robustness\_improvement.py](https://github.com/WenhuiDeng2000/GLSN-structural-robustness/blob/main/src/country_robustness_improvement.py)" for calculating improvement rate in the structural robustness of individual countries based on MLL strategy under the two attack scenarios.</p><p>the script "[main.py](https://github.com/WenhuiDeng2000/GLSN-structural-robustness/blob/main/src/main.py)" for realizing all the calculations done by the above five scripts.</p>|
|output|After running a script, results will be saved in this folder. Please find out below How to Use.|
# **System Requirements**
**OS Requirements**

These scripts have been tested on *Windows10* operating system.

**Installing Python on Windows**

Before setting up the package, users should have Python version 3.8 or higher, and several packages set up from Python 3.8. The latest version of python can be downloaded from the official website: [https://www.python.org/](https://www.python.org/ ) 

**Hardware Requirements**

The package requires only a standard computer with enough RAM to support the operations defined by a user. For minimal performance, this will be a computer with about 4 GB of RAM. For optimal performance, we recommend a computer with the following specs:

RAM: 8+ GB
CPU: 4+ cores, 3.4+ GHz/core
# **Installation Guide**
**Package dependencies**

Users should install the following packages prior to running the code:

    matplotlib==3.4.2
    networkx==2.6.2
    numpy==1.21.1
    pandas==1.3.1
    pyechart== 1.9.0    

For a Windows10 operating system, users can install the packages as follows 

    pip install -r README.md 

If you want to install only one of the packages, use:

    pip install pandas==1.3.1
# **How to Use**
The script [main.py](https://github.com/WenhuiDeng2000/GLSN-structural-robustness/blob/main/src/main.py) is used to realize all the calculations done by the above five scripts. Open the *cmd* window in the folder “src”, then run the following command:

    python main.py

The results will be saved in a folder called "output".
# **Contact**
- Mengqiao Xu: stephanie1996@sina.com 
- Wenhui Deng: dengwenhui2021@163.com 

