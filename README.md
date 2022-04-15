
---

# ***Driple*** - Prediction of the Resource Consumption of Distributed Deep Learning Systems
- [***Driple*** - Prediction of the Resource Consumption of Distributed Deep Learning Systems](#driple---prediction-of-the-resource-consumption-of-distributed-deep-learning-systems)
  - [**Overview**](#overview)
  - [***Driple* inspector training**](#driple-inspector-training)
    - [**Environment setup**](#environment-setup)
    - [**Execute training**](#execute-training)
  - [**Training dataset generation**](#training-dataset-generation)
  - [**Reference**](#reference)

<img src="/structure.jpg" alt="*Driple* structure" width="500"/>


---
## **Overview**
Core implementation of *Driple* which is introduced in ACM SIGMETRICS 2022. Please refer to the following papers for more details.
 - Full paper
 - 2 page abstract


*Driple* trains a machine learning model, called *Driple* inspector, for predicting 12 metrics in terms of resource consumption. In particular, *Driple* predicts 1) burst duration, 2) idle duration, and 3) burst consumption for each 1) GPU utilization, 2) GPU memory utilization, 3) network TX thorughput, and 4) network RX throughput.

*Driple* applies two key designs:
  - **Graph nerual network (GNN)**: Machine learning libraries, such as TensorFlow and PyTorch, converts the given code for training into a computational graph. We take the graph as an input of *Driple* inspector to take a wide spectrum of training workloads.
  - **Transfer learning**: *Driple* inspector is built for each DT setting (i.e., number of PS and workers, network interconnect types, and GPU types) of a specific machine learning library (e.g., TensorFlow). We leverage transfer learning to reduce training time and dataset size required for training.

---
## ***Driple* inspector training**

The implementation for training *Driple* inspector is in `training`. In `training` you can find the following directories.
- `/training/driple`: python codes for executing training of inspectors.
- `/training/models`: Python implementation of GNN models, such as graph convolutional network (GCN), graph isomorphism network (GIN), message passing neural network (MPNN), and graph attention network (GAT).

*Driple* inspector can be trained with or without transfer learning. We provide the pre-trained model we use for transfer learning. Note that we provide the pre-trained model only for GCN algorithm.

### **Environment setup**
We implement and test the training part of *Driple* inspector in conda environment.
The dependencies and requirements of our conda setting is given in "driple_training_requirement.txt". You can set the similar conda environment through the followgin command.
```
conda install -n <env_name> driple_training_requirement.txt
```

### **Execute training**
To execute train, please follow the commands below. 

- Commands for training **without TL**
  - The command below is for training a model with the design choices (hyperparameters) best for *Driple*. You can easily change hyperparameters with the command line arguments.
  - Please enter the dataset for training for `--data`.
```
python3 -m driple.train.gcn --variable --gru --epochs=100000 --patience=1000 --variable_conv_layers=Nover2 --only_graph --hidden=64 --mlp_layers=3 --data=[Dataset].pkl 
```



- Commands for training **with TL**
  - To enable TL, add `--transfer`. Also, please specify the pre-trained model through `--pre-trained`.
  - For TL, you should set the hyperparameters identical to the pre-trained model.
```
python3 -m driple.train.gcn --variable --gru --epochs=100000 --patience=1000 --variable_conv_layers=Nover2 --only_graph --hidden=64 --mlp_layers=3 --pre_trained=training/pre-train.pkl --data=[Dataset].pkl --transfer
```



---
## **Training dataset generation**
To be updated soon.

---
## **Reference**
```
@article{corso2020principal,
  title={Principal Neighbourhood Aggregation for Graph Nets},
  author={Corso, Gabriele and Cavalleri, Luca and Beaini, Dominique and Li{\`o}, Pietro and Veli{\v{c}}kovi{\'c}, Petar},
  journal={arXiv preprint arXiv:2004.05718},
  year={2020}
}
```