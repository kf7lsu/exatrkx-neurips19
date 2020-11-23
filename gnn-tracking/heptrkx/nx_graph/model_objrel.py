from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt

OBJ_HIDDEN = 8
REL_HIDDEN = 8

class SegmentClassifier(snt.AbstractModule):

  def __init__(self, name="SegmentClassifier"):
    super(SegmentClassifier, self).__init__(name=name)

    self._obj_mlp = snt.Sequential([
      snt.nets.MLP([OBJ_HIDDEN, OBJ_HIDDEN, 3],
                   activation=tf.nn.relu,
                   activate_final=False),
    ])
  
    self._rel_mlp = snt.Sequential([
      snt.nets.MLP([REL_HIDDEN, REL_HIDDEN, REL_HIDDEN, 1],
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

    output_ops = []
    latent = self._first(input_op)
    #latent = utils_tf.concat([input_op, latent], axis=1)
    latent = self._first(latent)
    latent = self._second(latent)
    # Transforms the outputs into appropriate shapes.
    output_ops.append(latent)

    return output_ops

                         

