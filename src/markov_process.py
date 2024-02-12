## MODULES ##

import numpy as np
import networkx as nx
    

## MARKOV CHAINS ##
    
class WeightedGraph:
    def __init__(self):
        self.dict = {}

    def test_ergodicity(self):
        for (id,input) in self.dict.items():
            data = input['t']
            G = nx.DiGraph()

            for i in data.index:
                G.add_node(i)

                for j in data.columns:
                    w = data.at[i,j]

                    if w > 0:
                        transformed_w = -np.log(w)
                        G.add_edge(i,j,weight=transformed_w)

            irreducibility = nx.is_strongly_connected(G)
            aperiodicity = nx.is_aperiodic(G)

            self.dict[id]['irreducible'] = irreducibility
            self.dict[id]['aperiodic'] = aperiodicity
