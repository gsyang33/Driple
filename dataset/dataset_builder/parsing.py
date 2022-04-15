import tensorflow as tf
from google.protobuf import text_format


def import_pbtxt(file):
    with open(file) as f:
        pbtxt = f.read()
        graph_def = text_format.Parse(pbtxt, tf.compat.v1.GraphDef(), allow_field_number=1)
        return graph_def


def pbtxt_extract_op_node_name(p):
    graph_node_list = []
    for node in p.node:
        if "GPU" in node.device:
            graph_node_list.append(node)
    op_node_name_list_in_pbtxt = [node.name for node in graph_node_list]
    return op_node_name_list_in_pbtxt


def pbtxt_extract_op_node_edge(p):
    op_node_edge_list_in_pbtxt = []
    graph_node_list = []
    for node in p.node:
        if "GPU" in node.device:
            graph_node_list.append(node)
    graph_node_name_list = [node.name for node in graph_node_list]

    for node in graph_node_list:
        for input in node.input:
            if ":" in input:
                if input[:input.index(":")] in graph_node_name_list:
                    op_node_edge_list_in_pbtxt.append((input[:input.index(":")], node.name))
            elif "^" in input:
                if input[1:] in graph_node_name_list:
                    op_node_edge_list_in_pbtxt.append((input[1:], node.name))
            else:
                if input in graph_node_name_list:
                    op_node_edge_list_in_pbtxt.append((input, node.name))
    return op_node_edge_list_in_pbtxt


def pbtxt_extract_tensor_shape(p):
    with tf.compat.v1.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(p, name='')
        tensor_shape_in_pbtxt = {}

        tensor_does_not_exist_node = 0
        tensor_does_not_exist_node_op_list = []

        graph_node_list = []
        for node in p.node:
            if "GPU" in node.device:
                graph_node_list.append(node)
        for node in graph_node_list:
            try:
                tensor_shape_in_pbtxt[node.name] = sess.graph.get_tensor_by_name(node.name + ":0").shape
            except KeyError:
                tensor_does_not_exist_node += 1
                tensor_does_not_exist_node_op_list.append(node.op)

    tf.compat.v1.reset_default_graph()
    return tensor_shape_in_pbtxt


