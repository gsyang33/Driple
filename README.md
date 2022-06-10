# ***Driple*** 


- [***Driple***](#driple)
  - [**Overview**](#overview)
  - [***Driple* inspector training**](#driple-inspector-training)
    - [**Environment setup**](#environment-setup)
    - [**Execute training**](#execute-training)
  - [**Training dataset generation**](#training-dataset-generation)
    - [**Input and output feature records**](#input-and-output-feature-records)
    - [**Training dataset generation**](#training-dataset-generation-1)
  - [**Reference**](#reference)



<img src="https://raw.githubusercontent.com/gsyang33/driple/master/others/structure.jpg" alt="*Driple* structure" width="700"/>


---
## **Overview**
*Driple* is introduced in ACM SIGMETRICS 2022. Please refer to the following papers for more details.
 - [Full paper](https://doi.org/10.1145/3530895)
 - [2 page abstract](https://doi.org/10.1145/3489048.3530962)

*Driple* trains a machine learning model, called *Driple* inspector, for predicting 12 metrics in terms of resource consumption. In particular, *Driple* predicts 1) burst duration, 2) idle duration, and 3) burst consumption for each 1) GPU utilization, 2) GPU memory utilization, 3) network TX throughput, and 4) network RX throughput.

*Driple* applies two key designs:
  - **Graph neural network (GNN)**: Machine learning libraries, such as TensorFlow and PyTorch, converts the given code for training into a computational graph. We take the graph as an input of *Driple* inspector to take a broad spectrum of training workloads.
  - **Transfer learning**: *Driple* inspector is built for each DT setting (i.e., number of PS and workers, network interconnect types, and GPU types) of a specific machine learning library (e.g., TensorFlow). We leverage transfer learning to reduce the training time and dataset size required for training.

---
## ***Driple* inspector training**

The implementation for training *Driple* inspector is in `training`. In `training`, you can find the following directories.
- `/training/driple`: python codes for executing training of inspectors.
- `/training/models`: Python implementation of GNN models, such as graph convolutional network (GCN), graph isomorphism network (GIN), message passing neural network (MPNN), and graph attention network (GAT).

*Driple* inspector can be trained with or without transfer learning. We provide the pre-trained model we use for transfer learning. Note that we provide the pre-trained model only for GCN algorithm.

### **Environment setup**
We implement and test the training part of *Driple* inspector in conda environment.
The dependencies and requirements of our conda setting are given in "driple_training_requirement.txt". You can set a similar conda environment through the following command.
```
conda install -n <env_name> driple_training_requirement.txt
```

### **Execute training**
To execute training, please follow the commands below. 

- Command for training **without TL**
  - The command below is for training a model with the design choices (hyperparameters) best for *Driple*. You can easily change hyperparameters with the command line arguments.
  - Please enter the dataset for training for `--data`.
```
python3 -m driple.train.gcn --variable --gru --epochs=100000 --patience=1000 --variable_conv_layers=Nover2 --only_graph --hidden=64 --mlp_layers=3 --data=[Dataset].pkl 
```


- Command for training **with TL**
  - To enable TL, add `--transfer`. Also, please specify the pre-trained model through `--pre-trained`.
  - For TL, you should set the hyperparameters identical to the pre-trained model.
```
python3 -m driple.train.gcn --variable --gru --epochs=100000 --patience=1000 --variable_conv_layers=Nover2 --only_graph --hidden=64 --mlp_layers=3 --pre_trained=training/pre-train.pkl --data=[Dataset].pkl --transfer
```



---
## **Training dataset generation**

We first provide 14 datasets used in this paper (`/dataset/examples`). Look at "details of the dataset below" for checking the detailed DT setting that each dataset is built.

**<details><summary>Details of the dataset</summary>**


|          Name           |      GPU        | DP <br>topology   |   Network   | # of GPU<br>machines  |         Name          |      GPU      | DP <br>topology   | Network   | # of GPU<br>machines  |
|:---------------------:  |:-------------:  |:---------------:  |:----------: |:--------------------: |:--------------------: |:------------: |:---------------:  |:-------:  |:--------------------: |
|   V100-P1w2/ho-PCIe     |      V100       |   PS1/w2/homo     | Co-located  |           1           |  2080Ti-P4w4/he-40G   |    2080Ti     |  PS4/w4/hetero    |  40 GbE   |           2           |
|   V100-P2w2/ho-PCIe     |      V100       |   PS2/w2/homo     | Co-located  |           1           | TitanRTX-P4w4/he-40G  | Titan<br>RTX  |  PS4/w4/hetero    |  40 GbE   |           2           |
|  2080Ti-P1w2/ho-PCIe    |     2080Ti      |   PS1/w2/homo     | Co-located  |           1           |    V100-P5w5/he-1G    |     V100      |  PS5/w5/hetero    |  1 GbE    |           5           |
|  2080Ti-P1w3/ho-PCIe    |     2080Ti      |   PS1/w3/homo     | Co-located  |           1           |   2080Ti-P5w5/he-1G   |    2080Ti     |  PS5/w5/hetero    |  1 GbE    |           5           |
|  2080Ti-P2w2/he-PCIe    |     2080Ti      |  PS2/w2/hetero    | Co-located  |           1           |    V100-P5w5/he-1G    |     V100      |  PS5/w10/hetero   |  1 GbE    |           5           |
| TitanRTX-P2w2/he-PCIe   | Titan <br>RTX   |  PS2/w2/hetero    | Co-located  |           1           |  2080Ti-P5w10/he-1G   |    2080Ti     |  PS5/w10/hetero   |  1 GbE    |           5           |
|   2080Ti-P2w2/he-40G    |     2080Ti      |  PS2/w2/hetero    |   40 GbE    |           2           |                       |               |                   |           |                       |
|  TitanRTX-P2w2/he-40G   |  Titan<br>RTX   | PS2/w2/hetero     | 40 GbE      |           2           |                       |               |                   |           |                       |


</details>


The dataset consists of representative image classification and natural language processing models. We use tf_cnn_benchmark and OpenNMT for running the models. 

For developers who want to create their datasets, we provide an example of dataset generation below.


### **Input and output feature records**
To be updated soon.

### **Training dataset generation**

We convert computational graphs into adjacency and feature matrices. Also, we produce the training dataset composed of the converted matrices and output features.


- Command
  - Give the path for the resource consumption measurement by `--perf_result` parameter.
  - Enter the path to save the dataset to be created with `--save_path`, and the file name with `--dataset_name`.
  - To get more information about parameters, use `--help` option.
```
python3 dataset_builder/generate_dataset.py --perf_result=[Result].csv --batch_size=32 --num_of_groups=100 --num_of_graphs=320 --save_path=[Path] --dataset_name=[Dataset].pkl
```


---
## **Reference**

 - Gyeongsik Yang, Changyong Shin, Jeunghwan Lee, Yeonho Yoo, and Chuck Yoo. 2022. Prediction of the Resource Consumption of Distributed Deep Learning Systems. <i>Proc. ACM Meas. Anal. Comput. Syst.</i> 6, 2, Article 29 (June 2022), 25 pages. https://doi.org/10.1145/3530895
 - Gyeongsik Yang, Changyong Shin, Jeunghwan Lee, Yeonho Yoo, and Chuck Yoo. 2022. Prediction of the Resource Consumption of Distributed Deep Learning Systems. In <i>Abstract Proceedings of the 2022 ACM SIGMETRICS/IFIP PERFORMANCE Joint International Conference on Measurement and Modeling of Computer Systems</i> (<i>SIGMETRICS/PERFORMANCE '22</i>). Association for Computing Machinery, New York, NY, USA, 69–70. https://doi.org/10.1145/3489048.3530962
