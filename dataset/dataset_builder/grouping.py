from networkx.algorithms.community import asyn_fluidc


def node_grouping(g, n_of_groups):
    grouped_node_name_list = []
    grouped_node_idx = {}
    group_num = 0

    for i, community in enumerate(asyn_fluidc(g, k=n_of_groups, seed=42)):
        grouped_node_name_list.append(list(community))
        for node_name in list(community):
            grouped_node_idx[node_name] = group_num
        group_num += 1

    return grouped_node_name_list, grouped_node_idx














