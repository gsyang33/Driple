from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
from types import SimpleNamespace

import math
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from models.pytorch.gnn_framework import GNN
from driple.util.util import load_dataset, total_loss, total_loss_multiple_batches, \
    specific_loss_multiple_batches

import csv



def build_arg_parser():
    """
    :return:    argparse.ArgumentParser() filled with the standard arguments for a training session.
                    Might need to be enhanced for some train_scripts.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='../../', help='Data path.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--only_nodes', action='store_true', default=False, help='Evaluate only nodes labels.')
    parser.add_argument('--only_graph', action='store_true', default=False, help='Evaluate only graph labels.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.003, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=1000, help='Patience')
    parser.add_argument('--conv_layers', type=int, default=None, help='Graph convolutions')
    parser.add_argument('--variable_conv_layers', type=str, default='N', help='Graph convolutions function name')
    parser.add_argument('--mlp_layers', type=int, default=3, help='Fully connected layers in readout')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function to use.')
    parser.add_argument('--print_every', type=int, default=50, help='Print training results every')
    parser.add_argument('--transfer', action='store_true', default=False, help='Whether to use transfer learning on model training')
    parser.add_argument('--final_activation', type=str, default='LeakyReLu',
                        help='final activation in both mlp layers for nodes and S2S for Graph')
    parser.add_argument('--skip', action='store_true', default=False,
                        help='Whether to use the model with skip connections.')
    parser.add_argument('--gru', action='store_true', default=False,
                        help='Whether to use a GRU in the update function of the layers.')
    parser.add_argument('--fixed', action='store_true', default=False,
                        help='Whether to use the model with fixed middle convolutions.')
    parser.add_argument('--variable', action='store_true', default=False,
                        help='Whether to have a variable number of comvolutional layers.')
    parser.add_argument('--max_conv', type=int, default=1000, help='Maximum number of conv layers')
    parser.add_argument('--pre_trained', type=str, default='../../', help='Pre-trained model path.')   #yks
    return parser


# map from names (as passed as parameters) to function determining number of convolutional layers at runtime
VARIABLE_LAYERS_FUNCTIONS = {
    'N': lambda adj: adj.shape[1],
    'Nover2': lambda adj: adj.shape[1] // 2,
    '4log2N': lambda adj: int(4 * math.log2(adj.shape[1])),
    '3log2N': lambda adj: int(3 * math.log2(adj.shape[1])),
    '2log2N': lambda adj: int(2 * math.log2(adj.shape[1])),
    'log2N': lambda adj: int(math.log2(adj.shape[1])),
    '3sqrtN': lambda adj: int(3 * math.sqrt(adj.shape[1])),
    'log10N': lambda adj: int(math.log10(adj.shape[1])),
    'Nover3': lambda adj: adj.shape[1] // 3,
    'log30N': lambda adj: int(math.log(35, adj.shape[1]))
}




def execute_train(gnn_args, args):
    """
    :param gnn_args: the description of the model to be trained (expressed as arguments for GNN.__init__)
    :param args: the parameters of the training session
    """
    def re_load_data():
        adj, features, graph_labels = load_dataset(args.data, args.loss, args.only_nodes, args.only_graph,
                                                   print_baseline=False)
        features['train'] = [x.cuda() for x in features['train']]
        adj['train'] = [x.cuda() for x in adj['train']]
        graph_labels['train'] = [x.cuda() for x in graph_labels['train']]


    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = 'cuda' if args.cuda else 'cpu'
    print('Using device:', device)

    # load data
    adj, features, graph_labels = load_dataset(args.data, args.loss, args.only_nodes, args.only_graph,
                                                            print_baseline=False)

    # model and optimizer
    gnn_args = SimpleNamespace(**gnn_args)

    # compute avg_d on the training set
    if 'avg_d' in gnn_args.first_conv_descr['args'] or 'avg_d' in gnn_args.middle_conv_descr['args']:
        dlist = [torch.sum(A, dim=-1) for A in adj['train']]
        avg_d = dict(lin=sum([torch.mean(D) for D in dlist]) / len(dlist),
                     exp=sum([torch.mean(torch.exp(torch.div(1, D)) - 1) for D in dlist]) / len(dlist),
                     log=sum([torch.mean(torch.log(D + 1)) for D in dlist]) / len(dlist))
    if 'avg_d' in gnn_args.first_conv_descr['args']:
        gnn_args.first_conv_descr['args']['avg_d'] = avg_d
    if 'avg_d' in gnn_args.middle_conv_descr['args']:
        gnn_args.middle_conv_descr['args']['avg_d'] = avg_d

    gnn_args.transfer = args.transfer
    gnn_args.device = device
    gnn_args.nfeat = features['train'][0].shape[2]
    gnn_args.nodes_out = 0
    gnn_args.graph_out = graph_labels['train'][0].shape[-1]
    gnn_args.fc_layers = args.mlp_layers
    if gnn_args.variable:
        assert gnn_args.conv_layers is None, "If model is variable, you shouldn't specify conv_layers (maybe you " \
                                             "meant variable_conv_layers?) "
    else:
        assert gnn_args.conv_layers is not None, "If the model is not variable, you should specify conv_layers"
    gnn_args.conv_layers = VARIABLE_LAYERS_FUNCTIONS[
        args.variable_conv_layers] if gnn_args.variable else args.conv_layers
    model = GNN(**vars(gnn_args))
    if args.transfer is True:
        model.load_state_dict(torch.load(args.pre_trained), False)

    for name, param in model.named_parameters():
        if args.transfer is True:
            if "conv_layers.2" in name or "graph_read_out.mlp" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params", pytorch_total_params)

    def move_cuda(dset):
        assert args.cuda, "Cannot move dataset on CUDA, running on cpu"
        if features[dset][0].is_cuda:
            # already on CUDA
            return
        features[dset] = [x.cuda() for x in features[dset]]
        adj[dset] = [x.cuda() for x in adj[dset]]
        graph_labels[dset] = [x.cuda() for x in graph_labels[dset]]

    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("device:", device)
        model.cuda()

        # move train, val to CUDA (delay moving test until needed)
        move_cuda('train')
        move_cuda('val')

    def train(epoch):
        """
        Execute a single epoch of the training loop

        :param epoch:int the number of the epoch being performed (0-indexed)
        """
        t = time.time()

        # train step
        model.train()

        for batch in range(len(adj['train'])):
            optimizer.zero_grad()
            output = model(features['train'][batch], adj['train'][batch])

            loss_train = total_loss(output, graph_labels['train'][batch], loss=args.loss,
                                    only_nodes=args.only_nodes, only_graph=args.only_graph)
            torch.cuda.empty_cache()
            loss_train.backward()
            torch.cuda.empty_cache()
            optimizer.step()
            del output
            torch.cuda.empty_cache()

        # validation epoch
        model.eval()
        output_val = 0.0
        for batch in range(len(adj['val'])):
            output_temp = model(features['val'][batch], adj['val'][batch])
            loss_val_temp = total_loss(output_temp, graph_labels['val'][batch], loss=args.loss,
                                    only_nodes=args.only_nodes, only_graph=args.only_graph).tolist()
            output_val = output_val + loss_val_temp
            del output_temp
            del loss_val_temp
            torch.cuda.empty_cache()

        output_val = output_val / len(adj['val'])

        return loss_train.data.item(), output_val

    def compute_test():
        """
        Evaluate the current model on all the sets of the dataset, printing results.
        This procedure is destructive on datasets.
        """
        model.eval()

        f = open('output-{}-{}-{}-{}-{}-{}-{}-{}-{}.csv'.format(epoch, gnn_args.first_conv_descr['layer_type'], args.hidden, args.variable_conv_layers, args.mlp_layers, args.lr, args.weight_decay, args.dropout, args.data), 'w', encoding='utf-8', newline='')
        wr = csv.writer(f, delimiter=",")
        wr.writerow(["predict"]*12 + ["label"]*12)
        resource = [["MEM_util"]*3 + ["GPU_util"]*3 + ["Network_rx"]*3 + ["Network_tx"]*3]
        value_category = [["avg Idle time", "avg Active time", "avg Peak consumption"]*4]
        wr.writerow([y for x in resource for y in x]*2)
        wr.writerow([y for x in value_category for y in x]*2)

        sets = list(features.keys())
        for dset in sets:

            wr.writerow(dset)

            # move data on CUDA if not already on it
            if args.cuda:
                move_cuda(dset)

            output_val = 0.0
            for batch in range(len(adj[dset])):
                output_temp = model(features[dset][batch], adj[dset][batch])
                for i in range(len(graph_labels[dset][batch])):
                    save_csv = output_temp.tolist()[i] + graph_labels[dset][batch].tolist()[i]
                    wr.writerow(save_csv)
                loss_val_temp = total_loss(output_temp, graph_labels[dset][batch], loss=args.loss,
                                           only_nodes=args.only_nodes, only_graph=args.only_graph).tolist()
                output_val = output_val + loss_val_temp
                del output_temp
                del loss_val_temp
                torch.cuda.empty_cache()
            output_val = output_val / len(adj[dset])
            print("average_output_val:", output_val)

            print("Test set results ", dset, ": loss= {:.4f}".format(output_val))

            del output_val


            del features[dset]
            del adj[dset]
            del graph_labels[dset]
            torch.cuda.empty_cache()

        f.close()


    sys.stdout.flush()
    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = -1

    sys.stdout.flush()
    start = time.time()

    with tqdm(range(args.epochs), leave=False, unit='epoch') as t:
        for epoch in t:
            loss_train, loss_val = train(epoch)
            loss_values.append(loss_val)
            end = time.time()
            #print("epoch: {}, time: {:.4f}, loss.train: {:.9f}, loss.val: {:.9f}".format(epoch, end - start, loss_train, loss_val))
            t.set_description('loss.train: {:.9f}, loss.val: {:.9f}'.format(loss_train, loss_val))
            sys.stdout.flush()
            del loss_train
            del loss_val
            if loss_values[-1] < best:

                # save current model
                torch.save(model.state_dict(), '{}-{}-{}-{}-{}-{}-{}-{}-{}.pkl'.format(epoch, gnn_args.first_conv_descr['layer_type'], args.hidden, args.variable_conv_layers, args.mlp_layers, args.lr, args.weight_decay, args.dropout, args.data))

                # remove previous model
                if best_epoch >= 0:
                    os.remove('{}-{}-{}-{}-{}-{}-{}-{}-{}.pkl'.format(best_epoch, gnn_args.first_conv_descr['layer_type'], args.hidden, args.variable_conv_layers, args.mlp_layers, args.lr, args.weight_decay, args.dropout, args.data))
                # update training variables
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                print('Early stop at epoch {} (no improvement in last {} epochs)'.format(best_epoch + 1, bad_counter))
                break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch + 1))
    model.load_state_dict(torch.load('{}-{}-{}-{}-{}-{}-{}-{}-{}.pkl'.format(best_epoch, gnn_args.first_conv_descr['layer_type'], args.hidden, args.variable_conv_layers, args.mlp_layers, args.lr, args.weight_decay, args.dropout, args.data)))

    # Testing
    compute_test()
