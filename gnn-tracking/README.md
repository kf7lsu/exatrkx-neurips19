# Graph Neural Networks for particle track reconstruction

## Setup the environment
1. Follow the instructions in the top-level [README.md](https://github.com/exatrkx/exatrkx-work/blob/master/README.md)

2. After cloning or downloading the exatrkx repository, cd to the gnn-tracking directory and install the necessary dependencies
```
cd gnn-tracking
pip install -e .
```

## Example gnn-tracking to FPGA workflow

This project was used for [arXiv:1904.08986](https://arxiv.org/abs/1904.08986). 
Here we will try to reproduce the input dataset and training.


Our goal is to find the tracks in one event of the [kaggle trackml challenge](https://www.kaggle.com/c/trackml-particle-identification/data) 100 events [sample dataset](https://drive.google.com/open?id=1SGRIRIMDr1rpuB_m183Nvvx9uLSe8c9f).
This [detector description spreadsheet](https://drive.google.com/open?id=18zdZUXSqhy1KIywYkpw8a81ynNRg-DFF) will also be needed. 
The event [blacklist](https://drive.google.com/file/d/16_DsM0Vk1e3UlnjWgH7FLwldiEM3Nu_f/view?usp=sharing) is used for trackml scoring purposes and is optional.

Let's assume we will run all from the gnn-tracking directory. 
Let's further assume that you have downloaded all your data to the inputdata subdirectory (possibly a simlink).

Based on the settings, we will only be using this portion of the detector:
![TrackMLPixelDetector](https://user-images.githubusercontent.com/4932543/119066812-75a5d500-b995-11eb-9a7a-790cf9c1a69c.png)

Note there are two different models reported in the paper. 
The original interaction network (IN), which takes in 3 node features and 4 edge features.

![hls4ml_GNN](https://user-images.githubusercontent.com/4932543/119068618-645ec780-b999-11eb-99b7-7e32811accd2.png)

and the object-relational (objrel) model, which is a simpler GNN and takes in 3 node features and 1 null (zero) edge feature.

![OpenCL_GNN](https://user-images.githubusercontent.com/4932543/119068596-5e68e680-b999-11eb-807d-8f3ec3c1da81.png)


1. Create an output directory tree
```
mkdir out
mkdir out/hitgraphs_1_2GeV
mkdir out/nxgraphs_1_2GeV
mkdir out/segments_1_2GeV
mkdir out/nxgraphs_1_2GeV_objrel
mkdir out/segments_1_2GeV_objrel
```
2. Preprocess the hits to generate a hitsgraph structure

Check/edit the settings in [`configs/prep_big.yaml`](configs/prep_big.yaml). Then run

```
python scripts/prepare_LP_IN.py configs/prep_big.yaml --n-workers 16
```
[`prepare_LP_IN.py`](scripts/prepare_LP_IN.py) can be found in the `scripts` subdirectory.
The `prepare_LP_IN.py` script generates `2*n_phi_section*n_eta_sections` files per event (in compressed npy format).
`eventNNNNNNNN_gSSS_ID.npz` contains the IDs of all the hits in the event `NNNNNNNN` section `SSS`. 
`eventNNNNNNNN_gSSS.npz` contains the inputs to the GNN. 
The input [`Graph`](heptrkx/datasets/graph.py) object contains all "reasonable" doublet candidates, and it is structured as a `namedtuple (X, Ri, Ro, y)`
```
      X contains selected hit features (e.g. 'r', 'phi', 'z') normalized using the relevant feature_scale
      Ri is a matrix n_hits*n_edges mapping each hit to the "incoming" segment  ending on that hit
      Ro is a matrix n_hits*n_edges mapping each hit to the "outgoing" segment starting from that hit
      y is an array of booleans which are true if both hits on a segment were produced by the same particle
```

3. Convert histgraph to network graph used in graph_nets

Check/edit the job settings in [`configs/nxgraph_kaggle.yaml`](configs/nxgraph_kaggle.yaml). 
Then run either for the IN model:
```
hits_graph_to_tuple -b configs/nxgraph_kaggle.yaml
```
or for the objrel model:
```
hits_graph_to_tuple -b configs/nxgraph_kaggle_objrel.yaml --no-edge-feature
```
that converts the hitgraph npx files produces during the previous step into a networkx graph format used by deepmind's graph_nets library.
The [`hits_graph_to_tuple`](scripts/hits_graph_to_tuple) script generates `2*n_phi_section*n_eta_sections` files per event (in compressed npy
format).
`eventNNNNNNNN_gSSSSSSS_INPUT.npz` contains a graph the input node features (e.g position) and edge features (e.g. distance or null) for the event `NNNNNNNN` section `SSSSSSS`.
`eventNNNNNNNN_gSSSSSSS_TARGET.npz` contains the expected "solution" features for nodes and edges.

4. Train the GNN

Check/edit the job settings in [`configs/train_edge_classifier_kaggle_share.yaml`](configs/train_edge_classifier_kaggle_share.yaml). 
Then run (IN)
```
python train_nx_graph configs/train_edge_classifier_kaggle_share.yaml
```
or (objrel)
```
python train_nx_graph configs/train_edge_classifier_kaggle_share_objrel.yaml
```
You will see a series of printouts like these
```
# (iteration number), TD (get graph), TR (TF run)
# 00163, TD 26.2, TR 34.1
# 00311, TD 55.8, TR 63.7
...
```
showing the time it takes to reach each TF (tensorflow) checkpoint.

To check the performance of the GNN while it is being trained look at `out/segments_1_2GeV/v0_kaggle/big.log`
```
# (iteration number), T (elapsed seconds), Ltr (training loss), Precision, Recall
# 00163, T 61.0, Ltr 0.1000, Lge 0.1072, Precision 0.5259, Recall 0.5680
# 00311, T 120.9, Ltr 0.0956, Lge 0.0926, Precision 0.5767, Recall 0.6935
...
```

5. Evaluate the tracking performance

At any point you can run the script
```
evaluate_gnn_models configs/train_edge_classifier_kaggle_share.yaml 1000 evt1000_GNN_scores.pdf 
```
here `1000` is the event number to test on (with networkx input files like `out/nxgraphs_1_2GeV/event000001000_g000000000_INPUT.npz`). The optional argument
`--iteration 1652` instruct the script to look for a tensorflow checkpoint file like `out/segments_100/v0_kaggle/checkpoint_02638.ckpt.index`

The output should contain a printout like this that scores the double classifier performance:
```
Accuracy:            0.993767
Precision (purity):  0.957302
Recall (efficiency): 0.957947
```
and produce a `evt1000_GNN_scores.pdf` file with plots like these
![](docs/evt1000_GNN_scores.jpg)

6. Export the weights for use in [`hls4ml`](https://github.com/vesal-rm/hls4ml/tree/master/example-prjs/graph/)
Now you can run

```
evaluate_gnn_simple configs/train_edge_classifier_kaggle_share.yaml 
```

or

```
evaluate_gnn_simple_objrel configs/train_edge_classifier_kaggle_share_objrel.yaml
```


