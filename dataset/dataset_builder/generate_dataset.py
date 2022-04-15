import argparse
import save_graph


def main(args):
    perf_result = args.perf_result
    batch_size = args.batch_size
    num_of_groups = args.num_of_groups
    num_of_graphs = args.num_of_graphs
    save_path = args.save_path
    dataset_name = args.dataset_name
    save_graph.save_graph(perf_result,
                          batch_size,
                          num_of_groups,
                          num_of_graphs,
                          save_path,
                          dataset_name)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--perf_result', type=str, help='Path of DriplePerf result file.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size of the dataset. Default is 32.')
    parser.add_argument('--num_of_groups', type=int, default=100, help='Number of groups. Default is 100.')
    parser.add_argument('--num_of_graphs', type=int, default=320, help='Number of graphs in the dataset. Default is 320.')
    parser.add_argument('--save_path', type=str, help='Save path of the generated dataset.')
    parser.add_argument('--dataset_name', type=str, help='File name of the generated dataset.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_arguments()
    main(args)

