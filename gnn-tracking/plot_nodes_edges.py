import matplotlib.pyplot as plt
import numpy as np
import glob

n_edges = []
n_nodes = []
fracs = []
for f_name in glob.glob('./out/hitgraphs_100_2GeV/event00000*_g0*[0-9].npz'):
    f = np.load(f_name)
    n_edge = len(dict(f)['y'])
    n_node = len(dict(f)['X'])
    print(dict(f)['y'].shape)
    print(dict(f)['X'].shape)
    print(dict(f)['Ri_cols'].shape)
    print(dict(f)['Ri_rows'].shape)
    print(dict(f)['Ro_cols'].shape)
    print(dict(f)['Ro_rows'].shape)

    n_edges.append(n_edge)
    n_nodes.append(n_node)
    if n_edge>0:
        fracs.append(sum(dict(f)['y'])/n_edge)


plt.figure()
plt.hist(n_edges,bins=np.linspace(0,600,61))
plt.xlabel('number of edges')
plt.savefig('n_edges.png')

plt.figure()
plt.hist(n_nodes,bins=np.linspace(0,600,61))
plt.xlabel('number of nodes')
plt.savefig('n_nodes.png')


print("95% nodes", np.quantile(n_nodes, 0.95))
print("95% edges", np.quantile(n_edges, 0.95))

print("average fraction of truth edges", np.mean(fracs))
