import graph_nets as gn
import sonnet as snt
import networkx as nx
import numpy as np
import setGPU

import tensorflow as tf
tf.compat.v1.set_random_seed(42)
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

# consider only MAX_NODES nodes (randomly selected)
MAX_NODES = 130
X = input_graphs[0]['nodes']
Xmask = (X[:,0] > 0.4) & (np.abs(X[:,1]) < 0.2) # take 1/5 of 1/8 phi slice, and last 4 layers
X = X[Xmask]
#import matplotlib.pyplot as plt
#plt.figure()
#plt.hist(X[:,0],bins=np.linspace(0, 1.1, 101))
#plt.savefig('X.png')
#plt.figure()
#plt.hist(X[:,1],bins=np.linspace(-1.1, 1.1, 101))
#plt.savefig('Y.png')
index = np.sort(np.random.choice(X.shape[0], MAX_NODES, replace=False))
input_graphs[0]['nodes'] = X[index]
input_graphs[0]['n_node'] = np.asarray(input_graphs[0]['nodes'].shape[0])
# cut edges that don't connect the MAX_NODES nodes
mask = (np.isin(input_graphs[0]['receivers'],index)) & (np.isin(input_graphs[0]['senders'],index))
receivers = input_graphs[0]['receivers'][mask]
senders = input_graphs[0]['senders'][mask]
mapping = np.empty(index.max() + 1, dtype=index.dtype)
mapping[index] = np.arange(input_graphs[0]['nodes'].shape[0])
input_graphs[0]['receivers'] = mapping[receivers]
input_graphs[0]['senders'] = mapping[senders]
input_graphs[0]['edges'] = input_graphs[0]['edges'][mask]
input_graphs[0]['n_edge'] = np.asarray(input_graphs[0]['edges'].shape[0])
input_graphs[0]['globals'] = input_graphs[0]['globals']

print('{} nodes and {} edges'.format(input_graphs[0]['nodes'].shape[0], input_graphs[0]['edges'].shape[0]))

with np.load(file_name.replace("INPUT", "TARGET")) as f:
    target_graphs.append(dict(f.items()))

tf.reset_default_graph()
# make latent dim smaller (4)
LATENT = 4
encoder = gn.modules.GraphIndependent(
    edge_model_fn=lambda: snt.Sequential([snt.nets.MLP([LATENT, LATENT], activation=tf.nn.relu, activate_final=True)]),
    node_model_fn=lambda: snt.Sequential([snt.nets.MLP([LATENT, LATENT], activation=tf.nn.relu, activate_final=True)]),
    global_model_fn=None)

core = gn.modules.InteractionNetwork(
    edge_model_fn=lambda: snt.Sequential([snt.nets.MLP([LATENT, LATENT], activation=tf.nn.relu, activate_final=True)]),
    node_model_fn=lambda: snt.Sequential([snt.nets.MLP([LATENT, LATENT], activation=tf.nn.relu, activate_final=True)]),
    reducer=tf.unsorted_segment_sum)

decoder = gn.modules.GraphIndependent(
    edge_model_fn=lambda: snt.Sequential([snt.nets.MLP([LATENT, LATENT, LATENT, 1], activation=tf.nn.relu, activate_final=False), 
                                          tf.sigmoid]),
    node_model_fn=None,
    global_model_fn=None)

print("graph_net_module instantiated")
print()

# Create a placeholder using the first graph in the list as template.
graphs_tuple_ph = utils_tf.placeholders_from_data_dicts(input_graphs)
tf_output_graphs = decoder(core(encoder(graphs_tuple_ph)))

encoder_edge_biases = []
encoder_edge_weights = []
encoder_node_biases = []
encoder_node_weights = []

core_edge_weights = []
core_edge_biases = []
core_node_weights = []
core_node_biases = []

decoder_edge_biases = []
decoder_edge_weights = []

with tf.Session() as sess:
    # randomly initialize the weights
    init = tf.global_variables_initializer()
    sess.run(init)

    # get the output predictions
    feed_dict = utils_tf.get_feed_dict(graphs_tuple_ph, utils_np.data_dicts_to_graphs_tuple(input_graphs))
    output_graphs = sess.run(tf_output_graphs, feed_dict=feed_dict)

    # save the weights of the model
    for v in encoder._edge_model.trainable_variables:
        if 'b:0' in v.name:
            encoder_edge_biases.append(v.eval(session=sess))
        elif 'w:0' in v.name:
            encoder_edge_weights.append(v.eval(session=sess))
    for v in encoder._node_model.trainable_variables:
        if 'b:0' in v.name:
            encoder_node_biases.append(v.eval(session=sess))
        elif 'w:0' in v.name:
            encoder_node_weights.append(v.eval(session=sess))

    for v in core._edge_block._edge_model.trainable_variables:
        if 'b:0' in v.name:
            core_edge_biases.append(v.eval(session=sess))
        elif 'w:0' in v.name:
            core_edge_weights.append(v.eval(session=sess))
    for v in core._node_block._node_model.trainable_variables:
        if 'b:0' in v.name:
            core_node_biases.append(v.eval(session=sess))
        elif 'w:0' in v.name:
            core_node_weights.append(v.eval(session=sess))

    for v in decoder._edge_model.trainable_variables:
        if 'b:0' in v.name:
            decoder_edge_biases.append(v.eval(session=sess))
        elif 'w:0' in v.name:
            decoder_edge_weights.append(v.eval(session=sess))


output_graphs = utils_np.graphs_tuple_to_data_dicts(output_graphs)

print("input_graphs:")
print(input_graphs)
print("output_graphs:")
print(output_graphs)

nodes = input_graphs[0]['nodes']
edges = input_graphs[0]['edges']
receivers = input_graphs[0]['receivers']
senders = input_graphs[0]['senders']
predict = output_graphs[0]['edges']

# save the input and output graphs for use in test bench
np.savetxt('tb_input_node_features.dat',nodes.reshape((1,-1)),delimiter=' ')
np.savetxt('tb_input_edge_features.dat',edges.reshape((1,-1)),delimiter=' ')
np.savetxt('tb_receivers.dat',receivers.reshape((1,-1)),delimiter=' ',fmt='%d')
np.savetxt('tb_senders.dat',senders.reshape((1,-1)),delimiter=' ',fmt='%d')
np.savetxt('tb_output_edge_predictions.dat',predict.reshape((1,-1)),delimiter=' ')

import hls4ml
# save the weights for use in hls4ml
os.makedirs('./firmware/weights',exist_ok=True)
for i, (w, b) in enumerate(zip(encoder_edge_weights, encoder_edge_biases)):
    var = hls4ml.model.hls_model.WeightVariable('encoder_w%i'%i, type_name='ap_fixed<16,6>', precision='<16,6>', data=w)
    hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
    var = hls4ml.model.hls_model.WeightVariable('encoder_b%i'%i, type_name='ap_fixed<16,6>', precision='<16,6>', data=b)
    hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
for i, (w, b) in enumerate(zip(core_edge_weights, core_edge_biases)):
    var = hls4ml.model.hls_model.WeightVariable('core_w%i'%i, type_name='ap_fixed<16,6>', precision='<16,6>', data=w)
    hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
    var = hls4ml.model.hls_model.WeightVariable('core_b%i'%i, type_name='ap_fixed<16,6>', precision='<16,6>', data=b)
    hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
for i, (w, b) in enumerate(zip(decoder_edge_weights, decoder_edge_biases)):
    var = hls4ml.model.hls_model.WeightVariable('decoder_w%i'%i, type_name='ap_fixed<16,6>', precision='<16,6>', data=w)
    hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
    var = hls4ml.model.hls_model.WeightVariable('decoder_b%i'%i, type_name='ap_fixed<16,6>', precision='<16,6>', data=b)
    hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
