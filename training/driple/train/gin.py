from __future__ import division
from __future__ import print_function

from models.pytorch.gin.layer import GINLayer
from driple.util.train import execute_train, build_arg_parser

# Training settings
parser = build_arg_parser()
parser.add_argument('--gin_fc_layers', type=int, default=2, help='Number of fully connected layers after the aggregation.')
args = parser.parse_args()

execute_train(gnn_args=dict(nfeat=None,
                            nhid=args.hidden,
                            nodes_out=None,
                            graph_out=None,
                            dropout=args.dropout,
                            device=None,
                            first_conv_descr=dict(layer_type=GINLayer, args=dict(fc_layers=args.gin_mlp_layers)),
                            middle_conv_descr=dict(layer_type=GINLayer, args=dict(fc_layers=args.gin_mlp_layers)),
                            fc_layers=args.mlp_layers,
                            conv_layers=args.conv_layers,
                            skip=args.skip,
                            gru=args.gru,
                            fixed=args.fixed,
                            variable=args.variable),
                            transfer=args.transfer, args=args)
