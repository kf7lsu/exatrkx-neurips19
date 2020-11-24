#!/bin/bash
BATCHSIZE=1
SETUP="#import setGPU # 1 gpu
import os
os.environ['CUDA_VISIBLE_DEVICES']='' # no gpus
from graph_nets import utils_np, utils_tf
from heptrkx import load_yaml
from heptrkx.nx_graph import prepare
import glob
iteration = -1
batch_size = $BATCHSIZE*10
from heptrkx.postprocess.evaluate_tf import create_evaluator
config_file = 'configs/train_edge_classifier_kaggle_share.yaml'
config = load_yaml(config_file)
file_dir = config['make_graph']['out_graph']
hits_graph_dir = config['data']['input_hitsgraph_dir']
trk_dir = config['track_ml']['dir']
input_ckpt = os.path.join(config['segment_training']['output_dir'],config['segment_training']['prod_name'])
file_names = glob.glob(os.path.join(file_dir, 'event00000*_g0000000*_INPUT.npz'))
model, model_c, sess = create_evaluator(config_file, iteration, input_ckpt)
is_digraph = True
is_bidirection = False
generate_input_target = prepare.inputs_generator(file_dir, n_train_fraction=0.8)
I, T = generate_input_target(batch_size,is_train=False)"
echo $SETUP
for i in {0..9}
do
    j=$((BATCHSIZE*i))
    k=$((BATCHSIZE*i+BATCHSIZE))
    echo $i
    echo $j
    echo $k
    python -m timeit -s "$SETUP" -n 100 "model(utils_np.data_dicts_to_graphs_tuple(I[$j:$k]),utils_np.data_dicts_to_graphs_tuple(T[$j:$k]),use_digraph=is_digraph, bidirection=is_bidirection)"
done
