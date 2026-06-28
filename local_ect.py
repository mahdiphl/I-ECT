import torch
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset, Amazon, Reddit, WebKB, WikipediaNetwork, Actor, LINKXDataset, WikiCS, Coauthor
import numpy as np
from torch_geometric.utils import k_hop_subgraph
from matplotlib import pyplot

from layers.ect import EctLayer
from layers.config import EctConfig

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb


def compute_local_ect(dataset,
                      radius=1,
                      ECT_TYPE='points',
                      NUM_THETAS = 64,
                      DEVICE = 'cpu',
                      subsample_size=None
):
    '''
    dataset: pytorch geometric graph dataset
    radius: number of hops for the local graph neighborhoods (i.e. `k` in the `k_hop_subgraph` function)
    ECT_TYPE: type of structural information used for the ECT calculation; can be 'points', 'edges' or 'faces'
    NUM_THETAS: the approximation parameter for the resulting ECT; the computation outputs a NUM_THETAS*NUM_THETAS dimensional vector.
    DEVICE: device to be used for the computation
    subsample_size: number of randomly sampled nodes in `dataset` to compute the local ECT for; default is None which means that local ECT is determined for all nodes in the input graph.
    '''

    data = dataset[0]
    features = data.x
    if subsample_size != None:
        np.random.seed(42)
        idx = np.random.choice(
                     range(len(data.x)),
                     replace=False,
                     size=subsample_size,
                 )

        sub_nodes = np.array(range(len(data.x)))[idx]
    else:
        sub_nodes = np.array(range(len(data.x)))

    batch = Batch.from_data_list(
        [
            Data(x=(data.x)[k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[0]],
                 edge_index=k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[1])
            for i in list(sub_nodes)
        ]
    ).to(DEVICE)

    CONFIG = EctConfig(num_thetas=NUM_THETAS, bump_steps=NUM_THETAS,
                       normalized=True, device=DEVICE, num_features=features.shape[1], ect_type=ECT_TYPE)

    ectlayer = EctLayer(config=CONFIG)

    ect = ectlayer(batch)
    ect = ect.reshape(ect.shape[0], ect.shape[1] * ect.shape[2])
    return ect


def xgb_model(dataset,
              radius1=True,
              radius2=True,
              ECT_TYPE='points',
              NUM_THETAS = 64,
              DEVICE = 'cpu',
              metric='accuracy',
              subsample_size=None
):
    '''
    dataset: pytorch geometric graph dataset
    radius1: if True, compute local ECT w.r.t. 1-hop neighborhoods
    radius2: if True, compute local ECT w.r.t. 2-hop neighborhoods
    ECT_TYPE: type of structural information used for the ECT calculation; can be 'points', 'edges' or 'faces'
    NUM_THETAS: the approximation parameter for the resulting ECT; the computation outputs a NUM_THETAS*NUM_THETAS dimensional vector.
    DEVICE: device to be used for the computation
    metric: choose metric for the evaluation of the classification; can be either `accuracy` or `roc'.
    '''
    data = dataset[0]
    all_labels = data.y
    features = data.x
    try:
        if (len(data.train_mask.shape)>1)&(len(data.test_mask.shape)>1):
            train_mask = data.train_mask[:,0]
            test_mask = data.test_mask[:,0]
        elif (len(data.train_mask.shape)>1)&(not(len(data.test_mask.shape)>1)):
            train_mask = data.train_mask[:, 0]
            test_mask = data.test_mask
        else:
            train_mask = data.train_mask
            test_mask = data.test_mask
    except AttributeError:
        bool_list = [True,False]
        p = [.75,.25]
        train_mask = np.random.choice(bool_list,len(data.x),p=p)
        test_mask = [not x for x in train_mask]

    if subsample_size!=None:
        np.random.seed(42)
        idx = np.random.choice(
            range(len(data.x)),
            replace=False,
            size=subsample_size,
        )

        all_labels = all_labels[idx]
        features = features[idx]
        train_mask = train_mask[idx]
        test_mask = test_mask[idx]

    if radius1:
        ect = compute_local_ect(dataset,
                                 radius=1,
                                 ECT_TYPE=ECT_TYPE,
                                 NUM_THETAS=NUM_THETAS,
                                 DEVICE=DEVICE,
                                 subsample_size = subsample_size)
        ect_train = ect[train_mask]
        ect_test = ect[test_mask]

    if radius2:
        ect = compute_local_ect(dataset,
                              radius=2,
                              ECT_TYPE=ECT_TYPE,
                              NUM_THETAS=NUM_THETAS,
                              DEVICE=DEVICE,
                              subsample_size=subsample_size)
        ect_train_2 = ect[train_mask]
        ect_test_2 = ect[test_mask]

    sub_labels = np.array(all_labels)
    sub_features = features

    train = sub_features[train_mask]
    test = sub_features[test_mask]

    if (radius1 and radius2):
        train = torch.cat((ect_train, ect_train_2,train), 1)
        test = torch.cat((ect_test,ect_test_2,test),1)
    elif radius1:
        train = torch.cat((ect_train, train), 1)
        test = torch.cat((ect_test, test), 1)
    elif radius2:
        train = torch.cat((ect_train_2, train), 1)
        test = torch.cat((ect_test_2, test), 1)

    train_labels = sub_labels[train_mask]
    train_labels = torch.tensor(train_labels)
    test_labels = sub_labels[test_mask]
    test_labels = torch.tensor(test_labels)
    # Train an XGBoost model
    model = xgb.XGBClassifier()
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.fit_transform(test_labels)
    model.fit(train, train_labels)
    # Predict probabilities for the test set
    y_score = model.predict(test)
    print(f'Feature Importance: {model.feature_importances_}')
    # plot

    # pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    # pyplot.title('Feature Importances')
    # pyplot.show()
    if metric=='accuracy':
        acc = accuracy_score(test_labels, y_score)
        print(f'Accuracy: {acc:.4f}')
        return acc
    elif metric=='roc':
        roc = roc_auc_score(test_labels, y_score)
        print(f'ROC AUC: {roc:.4f}')
        return roc