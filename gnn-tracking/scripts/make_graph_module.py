import graph_nets as gn
import sonnet as snt
import networkx as nx
import numpy as np
import setGPU

import tensorflow as tf
from graph_nets import utils_np, utils_tf

from heptrkx import load_yaml
from heptrkx import master

import os
import glob
import argparse
import sys

parser = argparse.ArgumentParser(description='evalute GNN models')
add_arg = parser.add_argument
add_arg('config', help='configuration file for training')
add_arg('evtid', type=int, help='event ID')
add_arg('--iteration',  type=int, default=-1)
add_arg('--ckpt', default=None)

args = parser.parse_args()
evtid = args.evtid
iteration = args.iteration
input_ckpt = args.ckpt

config_file = args.config
config = load_yaml(config_file)
file_dir = config['make_graph']['out_graph']
hits_graph_dir = config['data']['input_hitsgraph_dir']
trk_dir = config['track_ml']['dir']
if input_ckpt is None:
    input_ckpt = os.path.join(config['segment_training']['output_dir'],
                              config['segment_training']['prod_name'])

file_names = [os.path.join(file_dir, "event000001000_g000000000_INPUT.npz")]
true_features = ['pt', 'particle_id', 'nhits']
batch_size = 1

n_batches = 1

event = master.Event(trk_dir, evtid)
hits = event.hits
truth = event.truth

all_graphs = []
is_digraph = True
is_bidirection = False
# evaluate each graph                                                                                                        

input_graphs = []
target_graphs = []
file_name = file_names[0]
with np.load(file_name) as f:
    input_graphs.append(dict(f.items()))

with np.load(file_name.replace("INPUT", "TARGET")) as f:
    target_graphs.append(dict(f.items()))

print("input_graphs:")
print(input_graphs)

tf.reset_default_graph()


graph_net_module = gn.modules.InteractionNetwork(
    edge_model_fn=lambda: snt.nets.MLP([32]),
    node_model_fn=lambda: snt.nets.MLP([32]))

print("graph_net_module instantiated")
print()

# Create a placeholder using the first graph in the list as template.
graphs_tuple_ph = utils_tf.placeholders_from_data_dicts(input_graphs[0:1])
tf_output_graphs = graph_net_module(graphs_tuple_ph)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    feed_dict = utils_tf.get_feed_dict(graphs_tuple_ph, utils_np.data_dicts_to_graphs_tuple(input_graphs[0:1]))
    output_graphs = sess.run(tf_output_graphs, feed_dict=feed_dict)

print("output_graphs")
print(output_graphs)


