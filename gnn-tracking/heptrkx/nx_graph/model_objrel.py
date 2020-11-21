from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt

class SegmentClassifier(snt.AbstractModule):

  def __init__(self, name="SegmentClassifier"):
    super(SegmentClassifier, self).__init__(name=name)

    self._obj_mlp = snt.Sequential([
      snt.nets.MLP([200, 200, 3],
                   activation=tf.nn.relu,
                   activate_final=False),
    ])
  
    self._rel_mlp = snt.Sequential([
      snt.nets.MLP([250, 250, 250, 1],
                   activation=tf.nn.relu,
                   activate_final=False),
    ])

    self._rel_sigmoid = snt.Sequential([tf.nn.sigmoid])

    with self._enter_variable_scope():
      self._second = modules.GraphIndependent(edge_model_fn=lambda: self._rel_sigmoid, 
                                              node_model_fn=None, 
                                              global_model_fn=None)



    self._first = modules.InteractionNetwork(
      edge_model_fn=lambda: self._rel_mlp,
      node_model_fn=lambda: self._obj_mlp,
      reducer=tf.unsorted_segment_sum
    )    

  def _build(self, input_op, num_processing_steps):

    print('input',input_op.edges.shape)
    print('input',input_op.nodes.shape)

    output_ops = []
    #for _ in range(num_processing_steps):
    latent = self._first(input_op)
    print('after first',latent.edges.shape)
    print('after first',latent.nodes.shape)
    #latent = utils_tf.concat([input_op, latent], axis=1)
    #print(latent.edges.shape)
    #print(latent.nodes.shape)
    latent = self._first(latent)
    print('after first again',latent.edges.shape)
    print('after first again',latent.nodes.shape)
    latent = self._second(latent)
    print('after second',latent.edges.shape)
    print('after second',latent.nodes.shape)
    # Transforms the outputs into appropriate shapes.
    output_ops.append(latent)

    return output_ops

                         

