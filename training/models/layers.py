import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}


def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()


class Set2Set(torch.nn.Module):
    r"""
    Set2Set global pooling operator from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper. This pooling layer performs the following operation

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Arguments
    ---------
        input_dim: int
            Size of each input sample.
        hidden_dim: int, optional
            the dim of set representation which corresponds to the input dim of the LSTM in Set2Set.
            This is typically the sum of the input dim and the lstm output dim. If not provided, it will be set to :obj:`input_dim*2`
        steps: int, optional
            Number of iterations :math:`T`. If not provided, the number of nodes will be used.
        num_layers : int, optional
            Number of recurrent layers (e.g., :obj:`num_layers=2` would mean stacking two LSTMs together)
            (Default, value = 1)
    """

    def __init__(self, nin, nhid=None, steps=None, num_layers=1, activation=None, device='cpu'):
        super(Set2Set, self).__init__()
        self.steps = steps
        self.nin = nin
        self.nhid = nin * 2 if nhid is None else nhid
        if self.nhid <= self.nin:
            raise ValueError('Set2Set hidden_dim should be larger than input_dim')
        # the hidden is a concatenation of weighted sum of embedding and LSTM output
        self.lstm_output_dim = self.nhid - self.nin
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.nhid, self.nin, num_layers=num_layers, batch_first=True).to(device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        Arguments
        ----------
            x: torch.FloatTensor
                Input tensor of size (B, N, D)

        Returns
        -------
            x: `torch.FloatTensor`
                Tensor resulting from the  set2set pooling operation.
        """

        batch_size = x.shape[0]
        n = self.steps or x.shape[1]

        h = (x.new_zeros((self.num_layers, batch_size, self.nin)),
             x.new_zeros((self.num_layers, batch_size, self.nin)))

        q_star = x.new_zeros(batch_size, 1, self.nhid)

        for i in range(n):
            # q: batch_size x 1 x input_dim
            q, h = self.lstm(q_star, h)
            # e: batch_size x n x 1
            e = torch.matmul(x, torch.transpose(q, 1, 2))
            a = self.softmax(e)
            r = torch.sum(a * x, dim=1, keepdim=True)
            q_star = torch.cat([q, r], dim=-1)

        return torch.squeeze(q_star, dim=1)


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:

    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
            (Default value = None)

    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer
    """

    def __init__(self, in_size, out_size, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None,
                 device='cpu'):
        super(FCLayer, self).__init__()

        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = nn.Linear(in_size, out_size, bias=bias).to(device)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        # ksyang: remove batch normalization
        #if b_norm:
        #    self.b_norm = nn.BatchNorm1d(out_size).to(device)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_size)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            if h.shape[1] != self.out_size:
                h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.b_norm(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'


class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(self, in_size, hidden_size, out_size, layers, mid_activation='relu', last_activation='none',
                 dropout=0., mid_b_norm=False, last_b_norm=False, device='cpu'):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(FCLayer(in_size, out_size, activation=last_activation, b_norm=last_b_norm,
                                                device=device, dropout=dropout))
        else:
            self.fully_connected.append(FCLayer(in_size, hidden_size, activation=mid_activation, b_norm=mid_b_norm,
                                                device=device, dropout=dropout))
            for _ in range(layers - 2):
                self.fully_connected.append(FCLayer(hidden_size, hidden_size, activation=mid_activation,
                                                    b_norm=mid_b_norm, device=device, dropout=dropout))
            self.fully_connected.append(FCLayer(hidden_size, out_size, activation=last_activation, b_norm=last_b_norm,
                                                device=device, dropout=dropout))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'


class GRU(nn.Module):
    """
        Wrapper class for the GRU used by the GNN framework, nn.GRU is used for the Gated Recurrent Unit itself
    """

    def __init__(self, input_size, hidden_size, device):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size).to(device)

    def forward(self, x, y):
        """
        :param x:   shape: (B, N, Din) where Din <= input_size (difference is padded)
        :param y:   shape: (B, N, Dh) where Dh <= hidden_size (difference is padded)
        :return:    shape: (B, N, Dh)
        """
        assert (x.shape[-1] <= self.input_size and y.shape[-1] <= self.hidden_size)

        (B, N, _) = x.shape
        x = x.reshape(1, B * N, -1).contiguous()
        y = y.reshape(1, B * N, -1).contiguous()

        # padding if necessary
        if x.shape[-1] < self.input_size:
            x = F.pad(input=x, pad=[0, self.input_size - x.shape[-1]], mode='constant', value=0)
        if y.shape[-1] < self.hidden_size:
            y = F.pad(input=y, pad=[0, self.hidden_size - y.shape[-1]], mode='constant', value=0)

        x = self.gru(x, y)[1]
        x = x.reshape(B, N, -1)
        return x


# class CNN(nn.Module):
#     def __init__(self, hidden_unit):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1)
#         # self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5, stride=1)
#         # self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(in_channels=5, out_channels=7, kernel_size=5, stride=1)
#
#         self.fc1 = nn.Linear(5*56*8, 32)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         # x = self.max1(x)
#         x = F.relu(self.conv2(x))
#         # x = self.max2(x)
#         # x = F.relu(self.conv3(x))
#         x = x.view(-1, 5*56*8)
#         x = F.relu(self.fc1(x))
#         return x


class S2SReadout(nn.Module):
    """
        Performs a Set2Set aggregation of all the graph nodes' features followed by a series of fully connected layers
    """

    def __init__(self, in_size, hidden_size, out_size, fc_layers=3, device='cpu', final_activation='relu'):
        super(S2SReadout, self).__init__()

        # set2set aggregation
        self.set2set = Set2Set(in_size, device=device)

        # fully connected layers
        self.mlp = MLP(in_size=2 * in_size, hidden_size=hidden_size, out_size=out_size, layers=fc_layers,
                       mid_activation="relu", last_activation=final_activation, mid_b_norm=True, last_b_norm=False,
                       device=device)

        # [cyshin]
        #self.cnn = CNN(in_size)

    # [cyshin] modified
    # def forward(self, x):
    #     # print(x.shape)
    #     # x = self.set2set(x)
    #     x = global_sort_pool(x, 64)
    #     # x = global_add_pool(x, 64)
    #     # print(x.shape)
    #     x = self.cnn(x)
    #     x = self.mlp(x)
    #     # print(x)
    #     return x

    # [cyshin] original code
    def forward(self, x):
        x = self.set2set(x)
        return self.mlp(x)


# [cyshin]
def global_sort_pool(x, k):
    r"""The global pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
    where node features are sorted in descending order based on their last
    feature channel. The first :math:`k` nodes form the output of the layer.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        k (int): The number of nodes to hold for each graph.

    :rtype: :class:`Tensor`
    """
    fill_value = x.min().item() - 1
    batch_x = x
    B, N, D = batch_x.size()

    _, perm = batch_x[:, :, -1].sort(dim=-1, descending=True)
    arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
    perm = perm + arange.view(-1, 1)

    batch_x = batch_x.view(B * N, D)
    batch_x = batch_x[perm]
    batch_x = batch_x.view(B, N, D)
    #print(batch_x.size())

    if N >= k:
        batch_x = batch_x[:, :k].contiguous()
    else:
        expand_batch_x = batch_x.new_full((B, k - N, D), fill_value)
        batch_x = torch.cat([batch_x, expand_batch_x], dim=1)

    #print("d", batch_x.size())

    batch_x[batch_x == fill_value] = 0
    x = batch_x.view(B, 1, k,  D)
    # x = batch_x.view(B, k * D)

    #print(x.size())

    return x

# [cyshin]
from torch_scatter import scatter
def global_add_pool(x, size):
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    batch = x
    size = int(batch.max().item() + 1) if size is None else size
    result = scatter(x, batch, dim=0, dim_size=size, reduce='add')
    return result