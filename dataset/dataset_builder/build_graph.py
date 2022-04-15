import networkx as nx
import pandas as pd
import parsing
import grouping


def nx_node_using_op_node_name(op_list):
    nx_node_form = []
    for op in op_list:
        nx_node_form.append((op, {"tensor_size":0, "node_type":0}))
    return nx_node_form


def nx_edge_using_op_node_edge(edge_list):
    return edge_list


def add_feature_from_pbtxt(g, pbtxt_data):
    tensor_shape_in_pbtxt = parsing.pbtxt_extract_tensor_shape(pbtxt_data)
    for op_node_name, tensor_shape in tensor_shape_in_pbtxt.items():
        tensor_size = 0
        try:
            for dim in tensor_shape:
                tensor_size = 1 if tensor_size == 0 else tensor_size
                try:
                    tensor_size *= dim
                except TypeError:
                    tensor_size = 0
        except ValueError:
            tensor_size = 0
            pass

        g.nodes[op_node_name]['tensor_size'] = tensor_size
    return


def add_feature_from_csv(g, node_type_file, pbtxt_data):
    node_type_dict = pd.read_csv(node_type_file, header=None, index_col=0, squeeze=True).to_dict()
    for node in pbtxt_data.node:
        if "GPU" in node.device:
            g.nodes[node.name]['node_type'] = node_type_dict[node.op]


def build_nx_graph(pbtxt_path, node_type_file, dnn_model):
    pbtxt_data = parsing.import_pbtxt(pbtxt_path + dnn_model + ".pbtxt")

    op_node_name = parsing.pbtxt_extract_op_node_name(pbtxt_data)
    op_node_edge = parsing.pbtxt_extract_op_node_edge(pbtxt_data)

    nx_node = nx_node_using_op_node_name(op_node_name)
    nx_edge = nx_edge_using_op_node_edge(op_node_edge)

    g = nx.MultiGraph()
    g.add_nodes_from(nx_node)
    g.add_edges_from(nx_edge)

    add_feature_from_pbtxt(g, pbtxt_data)
    add_feature_from_csv(g, node_type_file, pbtxt_data)
    return g


def group_nx_graph(g, n_of_groups):
    grouped_g = nx.Graph()
    grouped_op_node_name, op_node_group_num = grouping.node_grouping(g, n_of_groups)

    for group_num, group in enumerate(grouped_op_node_name):
        sub_g = nx.Graph()
        tensor_size_value = 0
        node_type_value = 0

        for op_node_name in group:
            node = [(op_node_name, g.nodes[op_node_name])]
            sub_g.add_nodes_from(node)
            tensor_size_value += int(g.nodes[op_node_name]['tensor_size'])
            node_type_value += float(g.nodes[op_node_name]['node_type'])

        grouped_g.add_node(group_num, tensorsize=tensor_size_value, n_of_grouped_nodes=len(group), node_type=node_type_value, nodesinfo=sub_g)

    op_edge_list = list(g.edges)
    for edge in op_edge_list:
        grouped_g.add_edge(op_node_group_num[edge[0]], op_node_group_num[edge[1]])

    return grouped_g


def print_all_node(g):
    print(list(g.nodes(data=True)))
    return
