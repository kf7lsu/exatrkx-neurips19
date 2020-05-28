from __future__ import absolute_import
import tensorflow as tf

from graph_nets import utils_tf
from graph_nets import utils_np

import yaml
import os
import numpy as np

from heptrkx.nx_graph.prepare import inputs_generator
from heptrkx.nx_graph import get_model, utils_data, utils_train, utils_io
from heptrkx.nx_graph.utils_io import ckpt_name
from heptrkx import load_yaml


def create_evaluator(config_name, iteration, input_ckpt=None):
    """
    @config: configuration for train_nx_graph
    """
    # load configuration file
    all_config = load_yaml(config_name)
    config = all_config['segment_training']
    config_tr = config['parameters']

    batch_size = n_graphs   = config_tr['batch_size']   # need optimization
    num_processing_steps_tr = config_tr['n_iters']      ## level of message-passing
    prod_name = config['prod_name']
    if input_ckpt is None:
        input_ckpt = os.path.join(config['output_dir'], prod_name)


    # generate inputs
    generate_input_target = inputs_generator(all_config['make_graph']['out_graph'], n_train_fraction=0.8)

    # build TF graph
    tf.compat.v1.reset_default_graph()
    model = get_model(config['model_name'])


    input_graphs, target_graphs = generate_input_target(n_graphs)
    input_ph  = utils_tf.placeholders_from_data_dicts(input_graphs, force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_data_dicts(target_graphs, force_dynamic_num_graphs=True)

    output_ops_tr = model(input_ph, num_processing_steps_tr)
    try:
        sess.close()
    except NameError:
        pass

    sess = tf.Session()
    saver = tf.train.Saver()
    if iteration < 0:
        saver.restore(sess, tf.train.latest_checkpoint(input_ckpt))
    else:
        saver.restore(sess, os.path.join(input_ckpt, ckpt_name.format(iteration)))

    def evaluator(input_graphs, target_graphs, use_digraph=False, bidirection=False):
        """
        input is graph tuples, sizes should match batch_size
        """
        feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
        predictions = sess.run({
            "outputs": output_ops_tr,
            "target": target_ph
        }, feed_dict=feed_dict)
        output = predictions['outputs'][-1]

        pred_node_names = []
        for gtuple in output_ops_tr:
            pred_node_names.append(gtuple.edges.name[:-2]) # remove ':0'
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
        tf.train.write_graph(constant_graph, './', 'constantgraph.pb', as_text=False)

        return utils_data.predicted_graphs_to_nxs(
            output, input_graphs, target_graphs,
            use_digraph=use_digraph,
            bidirection=bidirection)

    return evaluator, model, sess


def create_profiler(config_name, iteration, input_ckpt=None):
    """
    @config: configuration for train_nx_graph
    """
    # load configuration file
    all_config = load_yaml(config_name)
    config = all_config['segment_training']
    config_tr = config['parameters']

    batch_size = n_graphs   = config_tr['batch_size']   # need optimization
    num_processing_steps_tr = config_tr['n_iters']      ## level of message-passing
    prod_name = config['prod_name']
    if input_ckpt is None:
        input_ckpt = os.path.join(config['output_dir'], prod_name)

    # generate inputs
    generate_input_target = inputs_generator(all_config['make_graph']['out_graph'], n_train_fraction=0.8)

    # build TF graph
    tf.compat.v1.reset_default_graph()
    model = get_model(config['model_name'])

    input_graphs, target_graphs = generate_input_target(n_graphs)
    input_ph  = utils_tf.placeholders_from_data_dicts(input_graphs, force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_data_dicts(target_graphs, force_dynamic_num_graphs=True)

    output_ops_tr = model(input_ph, num_processing_steps_tr)

    try:
        sess.close()
    except NameError:
        pass

    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()
    if iteration < 0:
        saver.restore(sess, tf.train.latest_checkpoint(input_ckpt))
    else:
        saver.restore(sess, os.path.join(input_ckpt, ckpt_name.format(iteration)))

    def profiler(input_graphs, target_graphs, use_digraph=False, bidirection=False):
        """
        input is graph tuples, sizes should match batch_size
        """

        print(model._encoder._network._edge_model._build_function)
        print(model._encoder._network._node_model._build_function)

        for layer in model._core._edge_block._edge_model._layers:
            print(layer._output_sizes)
        for layer in model._core._node_block._node_model._layers:
            print(layer._output_sizes)

        print(model._decoder._edge_model._build_function)
        print(model._output_transform._edge_model._build_function)

        print(input_graphs.nodes.shape)
        print(input_graphs.edges.shape)
        print(input_graphs.receivers.shape)
        print(input_graphs.senders.shape)
        print(input_graphs.n_node.shape)
        print(input_graphs.n_edge.shape)        
        print(input_graphs.globals.shape)
        flops = 1

        #feed_dict = {input_ph: input_graphs, target_ph: target_graphs}

        #constant_graph = tf.Graph()
        #with constant_graph.as_default():
        #    tf.import_graph_def(constant_graph_def,name='') 
        #    op = constant_graph.get_operations()
        #    run_metadata = tf.compat.v1.RunMetadata()
        #    predictions = sess.run({
        #        "outputs": output_ops_tr,
        #       "target": target_ph
        #    }, feed_dict=feed_dict,
        #    options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE))
        #flops = tf.profiler.profile(constant_graph, options = tf.profiler.ProfileOptionBuilder.float_operation(),run_meta=run_metadata)

        return flops

    return profiler
