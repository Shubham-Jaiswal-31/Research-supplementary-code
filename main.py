!pip install rich==10.7.0
!pip install Pygments
!pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
!pip install pyg-lib torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
!pip install pytorch_lightning
!pip install graphein
!sudo apt-get install dssp

import torch
import pandas as pd
import os

# get our dataset
!git clone https://www.github.com/a-r-j/graphein

p_ds = 'graphein/datasets/pscdb/structural_rearrangement_data.csv'

df = pd.read_csv(p_ds)
print("size of database: ",df.size)

df.head()

display(df.loc[df['Free PDB'] == '1na5'])

# Remove all entries with label "other_motion"
df = df.loc[df["motion_type"] != "other_motion"]
print(df.motion_type.value_counts())

# Filter out pdbs that we have empirically found to cause errors downstream

bad_pdbs = ['1uc8', '1u6e', '3b7y', '1fdp', '2vj2', '1n2o', '1nxu', '1q18', '1pn2', '1vh3', '1xk7', '2q5r', '2q50', '2olu', '1w2f', '1wxx', '2zc2', '3bwb', '2qt8', '2amj', '1h7s', '1zvw', '2f9t', '2f4b', '1jfv', '1ujm', '2b4v', '2etv', '3c8v', '1tr9', '2imo', '3bgu', '2pyq', '1upl', '1x2g', '2gzr', '1od4', '2qvl', '2i7h', '2dg0', '1evq', '2oyc', '2fyo', '1kko', '2a0k', '1uuj', '1u8t', '2bl7', '2c3v', '1p5q', '2gvl', '2hvf', '2rad', '3c8n', '2z2n', '2a6c', '1b6w', '1u24', '2nyh', '1k75', '2r7f', '2raf', '2qgq', '1ww8', '3c8h', '2eh1', '3db4', '2q4o', '3eo4', '2hzr', '2ghr', '1zs7', '1g5r', '2r11', '1yqg', '3eua', '2dg0', '1evq', '2oyc', '2fyo', '2a0k', '1uj4', '1g7r', '1ynf', '1wpo', '2qm8', '1vk3', '2cn2', '2vcl', '2rkt', '1l7o', '2r4i', '2it9', '1n4d', '2g6v', '1i3c', '1ox8', '1fbt', '1n57', '2rem', '1xxl', '2v3j', '2g0a', '1q6h', '2q22'] \
          + ['1lfh', '1gqz', '1yz5', '1oid', '1z15', '1l5t', '2cgk', '1sbq', '1vgt', '2ps3', '1bet', '1r5d', '1eut', '1qmt', '1ogb', '2i78', '3b8s', '1uln', '1so7', '1kn9', '1pu5', '1rlr', '1sgz', '1k5h', '1kx9'] \
          + ['1tib', '1bsq', '1r1c', '1fo8', '1s0g', '1goh', '1m47', '1xyf', '1lyy', '2ofj', '1uzq', '2qfd', '1br8', '1h8n', '1o7v', '1pz9', '1a8d', '1yjl', '2uyr', '2plf', '2rck', '1pbg', '1pgs'] \
          + ['2i2w', '1beo', '1hxj', '1gy0', '2qzp', '2c7i', '2box', '1psr', '2yrf', '1mzl', '2fp8', '1arl', '2yyv', '1iad', '2qev', '2veo', '1l5n', '1g24', '1w90', '1u4r', '1xze', '1z52', '1jwf', '2dcz', '1slc'] \
          + ['2d1z', '1oee', '4kiv', '2cbm', '1glh', '1qsa', '1bk7', '1qjv', '1e9n', '1vz2', '1czt', '3bi9', '2e4p', '1o9z', '1l0x', '1qba', '2jcp', '1sgk', '2jh1', '2e1u', '1smn', '2fma', '2g7e', '2vl3', '1g0z'] \
          + ['1vh3', '1x0v', '2b4v', '1tr9', '3bgu', '3bu9', '1uj4', '3c8n', '2a6c', '2r7f', '2hzb', '2qgq'] \
          + ['2e2n', '2qrj', '1r0s', '1lr9', '1dcl', '3ezm', '3c3b', '2ddb', '2qpq', '1aja', '1pzt', '3seb', '2p52', '2ze4', '1rn4', '2bce', '2fjy', '1m0z', '1fcq', '2f6l', '2ok3', '2bis', '1ppn', '1h8u', '1l8f', '2uy2', '2egt', '1plu', '1dqg', '2j0a', '3cj1', '153l'] \
          + ['2cfo','1qzt', '2ghb', '3cze', '2q50', '2olu', '2qza', '2zc2', '2f9t', '2vrk', '2f6p', '1xe8', '2pgx', '1x2g', '2qvl', '2i7h', '1evq', '1sza', '2q04', '1u8t', '2c3v', '2hvf', '1wgc', '1u24', '1k75', '1twd', '3c8h', '2ghr', '1zs7', '1yqg'] \
          + ['1zty', '1rki', '3bqh', '2orx', '2fz6', '1dkl', '2f08', '1ozt', '2rca', '1avk', '1vf8', '1bt2', '1esc', '1wns', '2h74', '1nk1', '1xqv', '1lwb', '1pp3', '1knl', '2zg2', '2atb', '2d05', '1ogm', '1kuf']

df = df.loc[~df['Free PDB'].isin(bad_pdbs)]
print(df.motion_type.value_counts())

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from graphein.protein.utils import get_obsolete_mapping

# split data to train/valid/test
seed=1
np.random.seed(seed)

labels_onehot = pd.get_dummies(df.motion_type)
train_val, test, y_train_val, y_test = train_test_split(df["Free PDB"], labels_onehot, random_state=seed, stratify=labels_onehot, test_size=0.2)
train, valid = train_test_split(train_val, random_state=seed, stratify=y_train_val, test_size=0.2)
train = train.sort_index(); valid = valid.sort_index(); test = test.sort_index()

# CONFIGS
import graphein.protein as gp
from functools import partial
from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.edges.distance import (add_peptide_bonds,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,
                                             add_ionic_interactions,
                                             add_aromatic_interactions,
                                             add_aromatic_sulphur_interactions,
                                             add_cation_pi_interactions
                                            )


# 1: Distance-based
dist_edge_func = \{"edge_construction_functions": [partial(gp.add_distance_threshold, threshold=5, long_interaction_threshold=0)]\}

# 2: Biochemical interactions, select set
select_edge_func = \{"edge_construction_functions": [add_peptide_bonds,
                                                    add_hydrogen_bond_interactions,
                                                    add_disulfide_interactions,
                                                    add_ionic_interactions,
                                                    gp.add_salt_bridges]\}

# 3: Biochemical interactions, expanded set
all_edge_func = \{"edge_construction_functions": [add_peptide_bonds,
                                                add_aromatic_interactions,
                                                add_hydrogen_bond_interactions,
                                                add_disulfide_interactions,
                                                add_ionic_interactions,
                                                add_aromatic_sulphur_interactions,
                                                add_cation_pi_interactions,
                                                gp.add_hydrophobic_interactions,
                                                gp.add_vdw_interactions,
                                                gp.add_backbone_carbonyl_carbonyl_interactions,
                                                gp.add_salt_bridges]\}

# A: Just one-hot encodings
one_hot = \{"node_metadata_functions" : [gp.amino_acid_one_hot]\}

# B: Selected biochemical features
all_graph_metadata = \{"graph_metadata_functions" : [gp.rsa,
                                                    gp.secondary_structure]\}
all_node_metadata = \{"node_metadata_functions" : [gp.amino_acid_one_hot,
                                                  gp.meiler_embedding,
                                                  partial(gp.expasy_protein_scale, add_separate=True)],
                     "dssp_config": gp.DSSPConfig()\}


config_1A = gp.ProteinGraphConfig(**\{**dist_edge_func, **one_hot\})
config_1B = gp.ProteinGraphConfig(**\{**dist_edge_func, **all_graph_metadata, **all_node_metadata\})

config_2A = gp.ProteinGraphConfig(**\{**select_edge_func, **one_hot\})
config_2B = gp.ProteinGraphConfig(**\{**select_edge_func, **all_graph_metadata, **all_node_metadata\})

config_3A = gp.ProteinGraphConfig(**\{**all_edge_func, **one_hot\})
config_3B = gp.ProteinGraphConfig(**\{**all_edge_func, **all_graph_metadata, **all_node_metadata\})

# Plotting
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph

g1 = construct_graph(config=config_1A, pdb_code=train.values[1])
g2 = construct_graph(config=config_2A, pdb_code=train.values[1])
g3 = construct_graph(config=config_3A, pdb_code=train.values[1])

p1 = plotly_protein_structure_graph(
    g1,
    colour_edges_by="kind",
    colour_nodes_by="degree",
    label_node_ids=False,
    plot_title="",
    node_size_multiplier=1
    )
p2 = plotly_protein_structure_graph(
    g2,
    colour_edges_by="kind",
    colour_nodes_by="degree",
    label_node_ids=False,
    plot_title="",
    node_size_multiplier=1
    )
p3 = plotly_protein_structure_graph(
    g3,
    colour_edges_by="kind",
    colour_nodes_by="degree",
    label_node_ids=False,
    plot_title="",
    node_size_multiplier=1
    )

p1.show(); p2.show(); p3.show()

def make_label_map(pdbs, onehots):
  label_map = \{\}
  for idx, pdb in enumerate(onehots):
    label_map[pdbs.iloc[idx]] = torch.tensor(onehots[idx])
  return label_map

train_labels_onehot = labels_onehot[labels_onehot.index.isin(train.index)].values.tolist()
train_label_map = make_label_map(train, train_labels_onehot)

valid_labels_onehot = labels_onehot[labels_onehot.index.isin(valid.index)].values.tolist()
valid_label_map = make_label_map(valid, valid_labels_onehot)

test_labels_onehot = labels_onehot[labels_onehot.index.isin(test.index)].values.tolist()
test_label_map = make_label_map(test, test_labels_onehot)

from graphein.ml import InMemoryProteinGraphDataset, ProteinGraphDataset

# CHOOSE CONFIG FILE:
config = config_1A #1A is least memory-intensive
convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg", columns=["coords", "edge_index",
                                                                             "amino_acid_one_hot", "bulkiness",
                                                                             "meiler", "rsa", "pka_rgroup",
                                                                             "isoelectric_points", "polaritygrantham",
                                                                             "hphob_black", "transmembranetendency"])

train_ds = InMemoryProteinGraphDataset(
    root="data/",
    name="train",
    pdb_codes=train,
    graph_label_map=train_label_map,
    graphein_config=config,
    graph_format_convertor=convertor,
    graph_transformation_funcs=[],
    )

valid_ds = InMemoryProteinGraphDataset(
    root="data/",
    name="valid",
    pdb_codes=valid,
    graph_label_map=valid_label_map,
    graphein_config=config,
    graph_format_convertor=convertor,
    graph_transformation_funcs=[]
    )

test_ds = InMemoryProteinGraphDataset(
    root="data/",
    graph_label_map=test_label_map,
    name="test",
    pdb_codes=test,
    graphein_config=config,
    graph_format_convertor=convertor,
    graph_transformation_funcs=[]
    )

    from torch_geometric.loader import DataLoader

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=4, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=4, drop_last=False)

import pytorch_lightning as pl
import torch
import torch.nn as nn
import itertools
import torchmetrics

"""EGNN Implementation from Satorras et al. https://github.com/vgsatorras/egnn"""

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = [nn.Linear(hidden_nf, hidden_nf)]
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr

class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i, j in itertools.product(range(n_nodes), range(n_nodes)):
        if i != j:
            rows.append(i)
            cols.append(j)

    return [rows, cols]


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import global_add_pool
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import LabelBinarizer

def process_features(batch, just_onehot: bool, also_concat_coords=False):
  """
  Converts batches that are passed into the dataloader into a valid tensor of dimension (n_feats, n_nodes)
  """
  if just_onehot:
    h = batch.amino_acid_one_hot.float()
    return h

  all_feats = ["amino_acid_one_hot", "bulkiness",
               "meiler", "rsa", "pka_rgroup",
               "isoelectric_points", "polaritygrantham",
               "hphob_black", "transmembranetendency"]

  dssp_ss = ['H','B','E','G','I','T','S','-']
  lb = LabelBinarizer().fit(dssp_ss)

  h_list = []
  for attrname in all_feats:
    try:
      attr = getattr(batch, attrname)
    except:
      continue
    if type(attr[0]) == pd.Series: #  converting pd.Series to numpy
        for i in range(len(attr)):
          attr[i] = attr[i].to_numpy()
        attr_tens = torch.concat(attr).cuda()
        h_list.append(attr_tens)


    if attrname == 'ss': #  onehotting dssp , in case it is a string array
        att_onehot = torch.FloatTensor(lb.transform(attr[0])).cuda()
        h_list.append(att_onehot)
    else:
        if len(attr.shape) == 1:
          attr = attr.unsqueeze(1)
        h_list.append(attr.cuda())


  h = torch.concatenate(h_list, dim=1)

  return h.to(torch.float32).cuda()

  class SimpleEGNN(pl.LightningModule):
  def __init__(self,
               n_feats=20,
               hidden_dim=32,
               out_feats=32,
               edge_feats=0,
               n_layers=2,
               num_classes=6,
               batch_size=16,
               just_onehot=False,
               loss_fn = CrossEntropyLoss):

      super().__init__()
      self.num_classes = num_classes
      self.model = EGNN(
          in_node_nf=n_feats,
          out_node_nf=out_feats,
          in_edge_nf=edge_feats,
          hidden_nf=hidden_dim,
          n_layers=n_layers,
      )
      self.decoder = nn.Sequential(
          nn.Linear(out_feats, out_feats),
          nn.ReLU(),
          nn.Linear(out_feats, num_classes),
      )
      self.loss = loss_fn()
      self.just_onehot = just_onehot
      self.batch_size = batch_size
      self.training_step_outputs = []
      self.training_step_labels = []
      self.valid_step_outputs = []
      self.valid_step_labels = []
      self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)
      self.micro_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='micro')
      self.macro_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro')

  def log_metrics(self, y, preds, mode='train'):
    '''
    y:      True data labels
    preds:  Predicted data labels
    mode:   Either 'train' or 'test'. For logging
    '''
    preds = nn.functional.softmax(preds, dim=0)

    micro_f1 = self.micro_f1(preds, y)
    self.log(f'f1/\{mode\}/micro_f1', micro_f1)
    macro_f1 = self.macro_f1(preds, y)
    self.log(f'f1/\{mode\}/macro_f1', macro_f1)

    accuracy = self.accuracy(torch.argmax(preds, dim=1), y)
    self.log(f'accuracy/\{mode\}', accuracy)
    for label in torch.unique(y):
      y_sub = y[torch.where(y == label)]
      pred_sub = preds[torch.where(y == label)]
      class_acc = self.accuracy(torch.argmax(pred_sub, dim=1), y_sub)
      self.log(f'class_acc/\{mode\}/accuracy_\{label\}', class_acc)

  def configure_loss(self, name: str):
      """Return the loss function based on the config."""
      return self.loss

  # --- Forward pass
  def forward(self, x):
      '''
      x.aa = torch.cat([torch.tensor(a) for a in x.amino_acid_one_hot]).float().cuda()
      x.c = torch.cat([torch.tensor(a).squeeze(0) for a in x.coords]).float().cuda()
      feats, coords = self.model(
          h=x.aa,
          x=x.c,
          edges=x.edge_index,
          edge_attr=None,
      )
      '''
      feats, coords = self.model(
          h=process_features(x, just_onehot=self.just_onehot),
          x=x.coords.float(),
          edges=x.edge_index,
          edge_attr=None,
      )

      feats = global_add_pool(feats, x.batch)
      return self.decoder(feats)

  def training_step(self, batch: Data, batch_idx) -> torch.Tensor:
      x = batch
      y = batch.graph_y.reshape((int(x.graph_y.shape[0] / self.num_classes), self.num_classes)).float()

      y_hat = self(x).float()
      _, y = y.max(dim=1)

      self.training_step_outputs.append(y_hat)
      self.training_step_labels.append(y)
      loss = self.loss(y_hat, y)

      return loss

  def on_train_epoch_end(self):
      all_preds = torch.concatenate(self.training_step_outputs, dim=0)
      all_labels = torch.concatenate(self.training_step_labels, dim=0)

      loss = self.loss(all_preds, all_labels)
      self.log('loss/train_loss', loss)
      self.log_metrics(all_labels, all_preds, 'train')

      self.training_step_outputs.clear()
      self.training_step_labels.clear()


  def validation_step(self, batch, batch_idx):
      x = batch
      y = batch.graph_y.reshape((int(x.graph_y.shape[0] / self.num_classes), self.num_classes)).float()
      _, y = y.max(dim=1)

      y_hat = self(batch).float()
      self.valid_step_outputs.append(y_hat)
      self.valid_step_labels.append(y)
      loss = self.loss(y_hat, y)
      return loss

  def on_validation_epoch_end(self):
      all_preds = torch.concatenate(self.valid_step_outputs, dim=0)
      all_labels = torch.concatenate(self.valid_step_labels, dim=0)

      loss = self.loss(all_preds, all_labels)
      self.log('loss/valid_loss', loss)
      self.log_metrics(all_labels, all_preds, 'valid')

      self.valid_step_outputs.clear()
      self.valid_step_labels.clear()


  def test_step(self, batch, batch_idx):
      x = batch
      # y = batch.graph_y.unsqueeze(1).float()
      y = batch.graph_y.reshape((int(x.graph_y.shape[0] / self.num_classes), self.num_classes)).float()

      y_hat = self(x).float()
      _, y = y.max(dim=1)

      loss = self.loss(y_hat, y)

      y_pred_softmax = torch.log_softmax(y_hat, dim = 1)
      y_pred_tags = torch.argmax(y_pred_softmax, dim = 1)
      self.log("test_loss", loss, batch_size=self.batch_size)
      return loss

  def configure_optimizers(self) -> torch.optim.Optimizer:
      return torch.optim.Adam(params=self.parameters(), lr=0.001)

      def calc_num_feats(loader):
      num_feats = 0
      for batch in loader:
        processed = process_features(batch, just_onehot=False)
        num_feats = processed.shape[1]
        break
      return num_feats

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    benchmark=True,
    deterministic=False,
    num_sanity_val_steps=0,
    max_epochs=400,
    log_every_n_steps=1
)

model = SimpleEGNN(n_feats=calc_num_feats(train_loader),
                  hidden_dim=32,
                  out_feats=32,
                  edge_feats=0,
                  n_layers=2,
                  num_classes=6,
                  batch_size=4,
                  just_onehot=False,
                  loss_fn = CrossEntropyLoss)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

trainer.test(model, test_loader)

%load_ext tensorboard
%tensorboard --bind_all --logdir "/content/lightning_logs" --port=6009
