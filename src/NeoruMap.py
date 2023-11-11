# 
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# prepare the data
# read in data
data = pd.read_csv("../resource/train_data2.csv")

# load in additional features for each neuron
feature_weights = pd.read_csv("../resource/feature_weights.csv")
morph_embeddings = pd.read_csv("../resource/morph_embeddings.csv")

# join all feature_weights
feature_weights["feature_weights"] = (
    feature_weights.filter(regex="feature_weight_")
    .sort_index(axis=1)
    .apply(lambda x: np.array(x), axis=1)
)
# delete the feature_weight_i columns
feature_weights.drop(
    feature_weights.filter(regex="feature_weight_").columns, axis=1, inplace=True
)

# join all morph_embed_i columns into a single np.array column
morph_embeddings["morph_embeddings"] = (
    morph_embeddings.filter(regex="morph_emb_")
    .sort_index(axis=1)
    .apply(lambda x: np.array(x), axis=1)
)
# delete the morph_embed_i columns
morph_embeddings.drop(
    morph_embeddings.filter(regex="morph_emb_").columns, axis=1, inplace=True
)

data = (
    data.merge(
        feature_weights.rename(columns=lambda x: "pre_" + x),
        how="left",
        validate="m:1",
        copy=False,
    )
    .merge(
        feature_weights.rename(columns=lambda x: "post_" + x),
        how="left",
        validate="m:1",
        copy=False,
    )
    .merge(
        morph_embeddings.rename(columns=lambda x: "pre_" + x),
        how="left",
        validate="m:1",
        copy=False,
    )
    .merge(
        morph_embeddings.rename(columns=lambda x: "post_" + x),
        how="left",
        validate="m:1",
        copy=False,
    )
)
# extract only connected neuron
linkedDat = data[data['connected'] == True]

pairs1 = set()
pairs2 = {}
c = 0
for index, row in data.iterrows():
    pairs1.add((row['pre_nucleus_x'], row['pre_nucleus_y'], row['pre_nucleus_z']))
    pairs1.add((row['post_nucleus_x'], row['post_nucleus_y'], row['post_nucleus_z']))
for index, row in linkedDat.iterrows():
    tmp = (row['pre_nucleus_id'], row['post_nucleus_id'])
    if tmp in pairs2:
        c += 1
    else:
        pairs2[tmp] = (row['pre_nucleus_x'], row['post_nucleus_x'], row['pre_nucleus_y'], row['post_nucleus_y'],
                       row['pre_nucleus_z'], row['post_nucleus_z'])


def color_nodes(G):
    """Colors the nodes of a graph based on their output and input.

    Args:
        G: A networkx graph object.

    Returns:
        A dictionary mapping node IDs to colors.
    """

    colors = []
    for node in G.nodes():
        if G.out_degree(node) > 0:
            if G.in_degree(node) > 0:
                colors.append("purple")
            else:
                colors.append("red")
        else:
            colors.append("blue")
    return colors


# Map non-filter direct neuron
G1 = nx.DiGraph()
for i in pairs2:  # add all pre and post to map
    G1.add_node(i[0], XY=(pairs2[i][0], pairs2[i][2]), XZ=(pairs2[i][0], pairs2[i][4]))
    G1.add_node(i[1], XY=(pairs2[i][1], pairs2[i][3]), XZ=(pairs2[i][1], pairs2[i][5]))
    G1.add_edge(i[0], i[1])

x = []
y = []
z = []
for i in pairs1:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])
plt.figure()
plt.scatter(x, y, color='pink')
colors = color_nodes(G1)
XY = nx.get_node_attributes(G1, 'XY')
nx.draw(G1, pos=XY, node_color=colors, node_size=20)
plt.show()

plt.figure()
plt.scatter(x, z, color='pink')
colors = color_nodes(G1)
XZ = nx.get_node_attributes(G1, 'XZ')
nx.draw(G1, pos=XZ, node_color=colors, node_size=20)
plt.show()
