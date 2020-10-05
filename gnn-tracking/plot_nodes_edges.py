import matplotlib.pyplot as plt
import numpy as np
import glob
import tqdm
n_edges = []
n_nodes = []
fracs = []
for f_name in tqdm.tqdm(glob.glob('./out/hitgraphs_1_2GeV/event00000*_g0*[0-9].npz')):
    f = np.load(f_name)
    n_edge = len(dict(f)['y'])
    n_node = len(dict(f)['X'])

    n_edges.append(n_edge)
    n_nodes.append(n_node)
    if n_edge>0:
        fracs.append(sum(dict(f)['y'])/n_edge)


plt.figure()
plt.hist(n_edges,bins=np.linspace(0,300,61))
plt.xlabel('number of edges')
plt.savefig('n_edges.png')

plt.figure()
plt.hist(n_nodes,bins=np.linspace(0,300,61))
plt.xlabel('number of nodes')
plt.savefig('n_nodes.png')


print("mean nodes", np.mean(n_nodes))
print("mean edges", np.mean(n_edges))

print("std nodes", np.std(n_nodes))
print("std edges", np.std(n_edges))

print("95% nodes", np.quantile(n_nodes, 0.95))
print("95% edges", np.quantile(n_edges, 0.95))

print("average fraction of truth edges", np.mean(fracs))
