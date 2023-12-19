import numpy as np
import pandas as pd
import torch.nn
import torch.utils.data as tutils
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pickle as pkl
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from bound_computation_utils import *
import xgboost as xgb
from plotting_utils import plot_rollout_eval
from scipy.special import expit
import time

HIDDEN_SIZE = 5
hidden_size_pygivenxc1c2 = 10
hidden_size_pygivenxc1 = 10
hidden_size_pygivenxc2 = 30
hidden_size_pc1c2 = 50
hidden_size_px_givenc1c2 = 30


def lb_weightedCE(predictions, weights):
    preds = torch.log(weights) + torch.log(predictions)

    loss = - torch.sum(torch.exp(preds), dim=1)
    return torch.mean(loss)


class XGBModel:
    def __init__(self, input_size, target_size, max_depth=6, eta=0.3, n_epochs=6, objective='binary:logistic'):
        self.model = None
        self.input_size = input_size
        self.target_size = target_size
        self.param = {'max_depth': max_depth, 'eta': eta, 'eval_metric': 'auc', 'objective': objective,
                      'num_class': target_size, 'max_delta_step': 1, 'scale_pos_weight': 1, 'n_estimators': 20}
        self.n_epochs = n_epochs

    def fit(self, X, y, Xtest, ytest):
        dtrain = xgb.DMatrix(X, label=y)
        dtest = xgb.DMatrix(Xtest, label=ytest)
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        result_dict = {}
        self.model = xgb.train(params=self.param, dtrain=dtrain, num_boost_round=self.n_epochs, evals=evallist,
                               early_stopping_rounds=self.n_epochs, evals_result=result_dict)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pkl.load(f)
        return self.model


class NNmodel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, target_size):
        super(NNmodel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.target_size)
        alpha_0 = torch.tensor(0.8)
        self.alpha = torch.nn.Parameter(alpha_0, requires_grad=True)

    def forward(self, x):
        hidden = self.fc1(x)
        act = self.act(hidden)
        op = self.fc2(act)
        return op


def train_multiclass_weightedCE(model, train_loader, valid_loader, optimizer, n_samples, n_epochs=30,
                                data_name='simulation',
                                cv=0, n_targets=2, model_name=''):
    print('Training black-box model on ', data_name)
    train_loss_trend = []
    test_loss_trend = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    if not os.path.exists("./results/"):
        os.mkdir("./results/")
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    if not os.path.exists(os.path.join("./ckpt/", data_name)):
        os.mkdir(os.path.join("./ckpt/", data_name))
    if not os.path.exists(os.path.join("./plots/", data_name)):
        os.mkdir(os.path.join("./plots/", data_name))
    if not os.path.exists(os.path.join("./results/", data_name)):
        os.mkdir(os.path.join("./results/", data_name))

    # alpha = alpha_0
    model.to(device)
    Xd = None
    for epoch in range(n_epochs):
        model.train()
        recall_train, precision_train, auc_train, correct_label_train, epoch_loss, count = 0, 0, 0, 0, 0, 0
        i = 0

        for i, (Xx, yy, weights) in enumerate(train_loader):
            Xd, label, wd = Xx.to(device), yy.to(device), weights.to(device)

            label = label.long()

            optimizer.zero_grad()
            predictions = torch.nn.Softmax(dim=1)(model(Xd))

            # label_onehot = label
            label_onehot = torch.nn.functional.one_hot(torch.flatten(label), num_classes=n_targets)

            _, predicted_label = predictions.max(1)
            pred_onehot = torch.nn.functional.one_hot(torch.flatten(predicted_label), num_classes=n_targets)

            reconstruction_loss = lb_weightedCE(predictions=predictions, weights=wd)
            auc, recall, precision, correct, auc_list, fpr_dict, tpr_dict, thresh_dict = evaluate_multiclass(
                label_onehot, pred_onehot, predictions)

            correct_label_train += correct
            auc_train += auc
            recall_train += recall
            precision_train += precision
            count += 1

            epoch_loss += reconstruction_loss.item()
            reconstruction_loss.backward()
            optimizer.step()

        test_loss, recall_test, precision_test, auc_test, correct_label_test, fpr_dict, tpr_dict, thresh_dict = \
            test_model_weightedCE(model, valid_loader, data_name=data_name, n=n_samples, cv=cv,
                                  d=Xd.shape[1], n_targets=n_targets)
        train_loss_trend.append(epoch_loss / (i + 1))
        test_loss_trend.append(test_loss)

        if epoch % 1 == 0:
            print('\nEpoch %d' % epoch)
            print('Training ===>loss: ', epoch_loss / (i + 1),
                  ' Accuracy: %.2f percent' % (100 * correct_label_train / (len(train_loader.dataset))),
                  ' AUC: %.2f' % (auc_train / (i + 1)))
            print('Test ===>loss: ', test_loss,
                  ' Accuracy: %.2f percent' % (100 * correct_label_test / (len(valid_loader.dataset))),
                  ' AUC: %.2f' % auc_test)

    test_loss, recall_test, precision_test, auc_test, correct_label_test, fpr_dict, tpr_dict, thresh_dict = \
        test_model_weightedCE(model, valid_loader, data_name=data_name, n=n_samples, cv=cv, d=Xd.shape[1],
                              n_targets=n_targets)
    print('Test AUC: ', auc_test)

    # Save model and results
    torch.save(model.state_dict(),
               './ckpt/' + data_name + '/%s_n_%d_d_%d_cv_%d.pt' % (model_name, n_samples, Xd.shape[1], cv))

    _, ax = plt.subplots(1, 1)
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plt.savefig(os.path.join('./plots', data_name, 'train_loss_n_%d_d_%d_cv_%d.pdf' % (n_samples, Xd.shape[1], cv)),
                dpi=300,
                bbox_inches='tight', bbox_extra_artists=[])
    plt.close()

    with open(os.path.join('./results/', data_name, 'roc_n_%d_d_%d_cv_%d.pkl' % (n_samples, Xd.shape[1], cv)),
              'wb') as f:
        pkl.dump({'fpr': fpr_dict, 'tpr': tpr_dict, 'thresh': thresh_dict}, f)


def train_wrapper(X, targets, lower_bounds, data_name, n_targets=2, loss_type='weighted_cross_entropy', **kwargs):
    n = X.shape[0]
    n_train = int(0.8 * X.shape[0])

    if 'cv' in kwargs.keys():
        cv = kwargs['cv']
    else:
        cv = 0

    if 'model_name' in kwargs.keys():
        model_name = kwargs['model_name']
    else:
        model_name = ''

    if 'train_idx' in kwargs.keys():
        train_idx = kwargs['train_idx']
        valid_idx = np.setdiff1d(np.arange(n), train_idx)
    else:
        train_idx = None
        valid_idx = None

    if train_idx is None:
        if 'f' in kwargs.keys():
            f = kwargs['f']
            kf = int(0.8 * f)
            train_idx = [i for i in range(n) if i % f <= kf]
            valid_idx = [i for i in range(n) if i % f > kf]
        else:
            kf = StratifiedShuffleSplit(n_splits=2, test_size=0.3)
            train_idx, valid_idx = list(kf.split(X, targets))[0]

    if 'batch_size' in kwargs.keys():
        batch_size = kwargs['batch_size']
    else:
        batch_size = 32

    if 'hidden_size' in kwargs.keys():
        hidden_size = kwargs['hidden_size']
    else:
        hidden_size = HIDDEN_SIZE

    if 'learning_rate' in kwargs.keys():
        learning_rate = kwargs['learning_rate']
    else:
        learning_rate = 0.01

    if 'n_epochs' in kwargs.keys():
        n_epochs = kwargs['n_epochs']
    else:
        n_epochs = 15

    model = None
    # proposed method
    if data_name == 'synthetic' or data_name == 'synthetic_continuous' or data_name == "synthetic_no_bounds" or data_name == 'IST':

        train_dataset = tutils.TensorDataset(torch.Tensor(X[train_idx, :]),
                                             torch.Tensor(targets[train_idx, :]),
                                             torch.Tensor(lower_bounds[train_idx]))
        valid_dataset = tutils.TensorDataset(torch.Tensor(X[valid_idx, :]),
                                             torch.Tensor(targets[valid_idx, :]),
                                             torch.Tensor(lower_bounds[valid_idx]))

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=len(X) - len(train_idx))

        model = NNmodel(input_size=X.shape[1], hidden_size=hidden_size, target_size=n_targets)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if loss_type == "weighted_cross_entropy":
            train_multiclass_weightedCE(model=model, train_loader=train_loader, valid_loader=valid_loader,
                                        optimizer=optimizer, n_samples=n,
                                        data_name=data_name, n_epochs=n_epochs, n_targets=n_targets, cv=cv,
                                        model_name=model_name)
        else:
            ValueError("%s Loss type not supported" % loss_type)

    elif data_name == "Logistic":
        train_dataset = tutils.TensorDataset(torch.Tensor(X[train_idx, :]),
                                             torch.Tensor(targets[train_idx, :]),
                                             torch.Tensor(np.ones((len(train_idx), targets.shape[1]))))
        label = np.expand_dims(np.argmax(lower_bounds[valid_idx], axis=1), axis=1)
        label_onehot = OneHotEncoder(sparse=False).fit_transform(label)
        valid_dataset = tutils.TensorDataset(torch.Tensor(X[valid_idx, :]),
                                             torch.Tensor(label_onehot),
                                             torch.Tensor(np.ones((len(valid_idx), targets.shape[1]))))

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=len(X) - int(0.8 * n_train))

        model = NNmodel(input_size=X.shape[1], hidden_size=5, target_size=targets.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model = train_multiclass(model=model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer,
                                 data_name=data_name, cv=cv)
    else:
        ValueError(' %s does not exist', data_name)

    return model


def train_wrapper_ist_c1c2(C1_data, C2_data, data_name, loss_type=None,
                           **kwargs):
    if loss_type is None:
        loss_type = ['binary_cross_entropy', 'mean_squared_error']
    n = C1_data.shape[0]

    if 'cv' in kwargs.keys():
        cv = kwargs['cv']
    else:
        cv = 0

    if 'train_idx' in kwargs.keys():
        train_idx = kwargs['train_idx']
        valid_idx = kwargs['valid_idx']
    else:
        train_idx = None
        valid_idx = None

    if train_idx is None:
        if 'f' in kwargs.keys():
            f = kwargs['f']
            kf = int(0.8 * f)
            train_idx = [i for i in range(n) if i % f <= kf]
            valid_idx = [i for i in range(n) if i % f > kf]
        else:
            if 'binary_cross_entropy' in loss_type:
                kf = StratifiedShuffleSplit(n_splits=2, test_size=0.3)
                train_idx, valid_idx = list(kf.split(C2_data, C1_data))[0]
            else:
                train_idx, valid_idx = train_test_split(np.arange(n), test_size=0.3)

    if 'batch_size' in kwargs.keys():
        batch_size = kwargs['batch_size']
    else:
        batch_size = 32

    if 'n_epochs' in kwargs.keys():
        n_epochs = kwargs['n_epochs']
    else:
        n_epochs = 80

    if 'hidden_size' in kwargs.keys():
        hidden_size = kwargs['hidden_size']
    else:
        hidden_size = 20

    if 'learning_rate' in kwargs.keys():
        learning_rate = kwargs['learning_rate']
    else:
        learning_rate = 0.001

    # This part learns a model for c1 given c2
    model_c1_givenc2 = None
    train_dataset = tutils.TensorDataset(torch.Tensor(C2_data[train_idx, :]),
                                         torch.Tensor(C1_data[train_idx, :]))
    valid_dataset = tutils.TensorDataset(torch.Tensor(C2_data[valid_idx, :]),
                                         torch.Tensor(C1_data[valid_idx, :]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_idx))

    model_c1_givenc2 = NNmodel(input_size=C2_data.shape[1], hidden_size=hidden_size, target_size=C1_data.shape[1])
    optimizer = torch.optim.Adam(model_c1_givenc2.parameters(), lr=learning_rate)
    if C1_data.shape[1] == 2 and loss_type[0] == 'binary_cross_entropy':
        model_c1_givenc2 = train_multilabel(model=model_c1_givenc2, train_loader=train_loader,
                                            valid_loader=valid_loader,
                                            optimizer=optimizer,
                                            data_name=data_name + '_c1_givenc2', n_epochs=n_epochs, cv=cv)
    elif C1_data.shape[1] == 1 and loss_type[1] == 'mean_squared_error':
        model_c1_givenc2 = train_regression(model=model_c1_givenc2, train_loader=train_loader,
                                            valid_loader=valid_loader,
                                            optimizer=optimizer,
                                            data_name=data_name + '_c1_givenc2', n_epochs=n_epochs, cv=cv)
    else:
        ValueError(' loss type %s for c1 does not exist', loss_type)

    # This part learns a model for c2 given c1
    train_dataset = tutils.TensorDataset(torch.Tensor(C1_data[train_idx, :]),
                                         torch.Tensor(C2_data[train_idx, :]))
    valid_dataset = tutils.TensorDataset(torch.Tensor(C1_data[valid_idx, :]),
                                         torch.Tensor(C2_data[valid_idx, :]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_idx))

    model_c2_givenc1 = NNmodel(input_size=C1_data.shape[1], hidden_size=hidden_size, target_size=C2_data.shape[1])
    optimizer = torch.optim.Adam(model_c2_givenc1.parameters(), lr=learning_rate)
    if C2_data.shape[1] == 1 and loss_type[1] == 'mean_squared_error':
        model_c2_givenc1 = train_regression(model=model_c2_givenc1, train_loader=train_loader,
                                            valid_loader=valid_loader,
                                            optimizer=optimizer,
                                            data_name=data_name + '_c2_givenc1', n_epochs=n_epochs, cv=cv)
    else:
        ValueError(' loss type %s for c1 does not exist', loss_type)

    return model_c1_givenc2, model_c2_givenc1


def train_wrapper_bounds(X, targets, lower_bounds, data_name, n_targets=2, loss_type='cross_entropy',
                         **kwargs):
    if 'cv' in kwargs.keys():
        cv = kwargs['cv']
    else:
        cv = 0

    n = X.shape[0]
    n_train = int(0.8 * X.shape[0])

    if 'train_idx' in kwargs.keys():
        train_idx = kwargs['train_idx']
        valid_idx = np.setdiff1d(np.arange(n), train_idx)
    else:
        train_idx = None
        valid_idx = None

    if train_idx is None:
        if 'f' in kwargs.keys():
            f = kwargs['f']
            kf = int(0.8 * f)
            train_idx = [i for i in range(n) if i % f <= kf]
            valid_idx = [i for i in range(n) if i % f > kf]
        else:
            kf = StratifiedShuffleSplit(n_splits=2, test_size=0.3)
            train_idx, valid_idx = list(kf.split(X, targets))[0]

    if 'batch_size' in kwargs.keys():
        batch_size = kwargs['batch_size']
    else:
        batch_size = 32

    if 'hidden_size' in kwargs.keys():
        hidden_size = kwargs['hidden_size']
    else:
        hidden_size = HIDDEN_SIZE

    if 'learning_rate' in kwargs.keys():
        learning_rate = kwargs['learning_rate']
    else:
        learning_rate = 0.005

    if 'n_epochs' in kwargs.keys():
        n_epochs = kwargs['n_epochs']
    else:
        n_epochs = 80

    if 'model' in kwargs.keys():
        model = kwargs['model']
    else:
        model = 'NNModel'

    if 'max_depth' in kwargs.keys():
        max_depth = kwargs['max_depth']
    else:
        max_depth = 6

    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    if not os.path.exists("./results/"):
        os.mkdir("./results/")
    if not os.path.exists(os.path.join("./ckpt/", data_name + '_bounds')):
        os.mkdir(os.path.join("./ckpt/", data_name + '_bounds'))
    if not os.path.exists(os.path.join("./plots/", data_name + '_bounds')):
        os.mkdir(os.path.join("./plots/", data_name + '_bounds'))
    if not os.path.exists(os.path.join("./results/", data_name + '_bounds')):
        os.mkdir(os.path.join("./results/", data_name + '_bounds'))

    # proposed method
    if data_name == 'synthetic' or data_name == 'synthetic_continuous' or data_name == "synthetic_no_bounds":
        train_dataset = tutils.TensorDataset(torch.Tensor(X[train_idx, :]),
                                             torch.Tensor(targets[train_idx, :]),
                                             torch.Tensor(lower_bounds[train_idx]))
        valid_dataset = tutils.TensorDataset(torch.Tensor(X[valid_idx, :]),
                                             torch.Tensor(targets[valid_idx, :]),
                                             torch.Tensor(lower_bounds[valid_idx]))

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=len(X) - len(train_idx))

        model = NNmodel(input_size=X.shape[1], hidden_size=HIDDEN_SIZE, target_size=n_targets)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if loss_type == "weighted_cross_entropy":
            train_multiclass_weightedCE(model=model, train_loader=train_loader, valid_loader=valid_loader,
                                        optimizer=optimizer, n_samples=n,
                                        data_name=data_name, n_epochs=100, cv=cv)
        else:
            ValueError(' loss type %s not implemented', loss_type)
    elif "IST_bounds" in data_name or 'synthetic_pxgivenc' in data_name or 'synthetic_no_bounds_bounds' in data_name:
        targets = targets.reshape(-1, 1)
        targets_one_hot = np.zeros((targets.shape[0], n_targets))
        enc = OneHotEncoder(sparse=False)
        targets_one_hot[train_idx, :] = enc.fit_transform(targets[train_idx])
        targets_one_hot[valid_idx, :] = enc.transform(targets[valid_idx])
        if model == 'NNModel':
            train_dataset = tutils.TensorDataset(torch.Tensor(X[train_idx, :]),
                                                 torch.Tensor(targets_one_hot[train_idx, :]),
                                                 torch.Tensor(np.ones((len(train_idx), targets.shape[1]))))

            valid_dataset = tutils.TensorDataset(torch.Tensor(X[valid_idx, :]),
                                                 torch.Tensor(targets_one_hot[valid_idx, :]),
                                                 torch.Tensor(np.ones((len(valid_idx), targets.shape[1]))))

            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            valid_loader = DataLoader(valid_dataset, batch_size=len(X) - int(0.8 * n_train))

            model = NNmodel(input_size=X.shape[1], hidden_size=hidden_size, target_size=targets_one_hot.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model = train_multiclass(model=model, train_loader=train_loader, valid_loader=valid_loader,
                                     optimizer=optimizer,
                                     data_name=data_name + '_bounds', n_epochs=n_epochs, cv=cv)
        elif model == 'XGBoost':
            if targets_one_hot.shape[1] >= 2:
                objective = 'multi:softprob'
            else:
                objective = 'binary:logistic'
            model = XGBModel(input_size=X.shape[1], target_size=targets_one_hot.shape[1], max_depth=max_depth,
                             objective=objective)
            model.fit(X[train_idx, :], targets_one_hot[train_idx, 1], Xtest=X[valid_idx, :],
                      ytest=targets_one_hot[valid_idx, 1])
            model.save(os.path.join('ckpt', data_name + '_bounds', 'xXGBoostModel_%d.pkl' % cv))
        else:
            ValueError(' %s model not implemented', model)

    else:
        ValueError(' %s does not exist', data_name)
    return model


def lb_weighted_CEloss(predictions, targets, weights):
    # print(predictions.shape)
    m = torch.nn.LogSoftmax(dim=1)
    batch_size = predictions.shape[0]
    loss = - weights * targets.view(batch_size, -1) * m(predictions)
    return torch.mean(torch.sum(loss, dim=1))


def train_multiclass(model, train_loader, valid_loader, optimizer, n_epochs=30, data_name='simulation',
                     cv=0):
    print('Training black-box model on ', data_name)

    train_loss_trend = []
    test_loss_trend = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        recall_train, precision_train, auc_train, correct_label_train, epoch_loss, count = 0, 0, 0, 0, 0, 0
        i = 0
        for i, (Xx, yy, weights) in enumerate(train_loader):
            Xd, label, wd = Xx.to(device), yy.to(device), weights.to(device)

            optimizer.zero_grad()
            predictions = model(Xd)

            pred_onehot = torch.FloatTensor(label.shape[0], label.shape[1]).to(device)
            _, predicted_label = predictions.max(1)
            pred_onehot.zero_()
            pred_onehot.scatter_(1, predicted_label.view(-1, 1), 1)

            if data_name != 'Ours':
                reconstruction_loss = torch.nn.CrossEntropyLoss()(predictions, label)
                auc, recall, precision, correct, auc_list, fpr_dict, tpr_dict, thresh_dict = evaluate_multiclass(
                    label, pred_onehot, predictions)
            else:
                label_onehot = torch.FloatTensor(label.shape[0], 2).to(device)
                label_onehot.zero_()
                _, wd_max = wd.max(1)
                label_onehot.scatter_(1, wd_max.view(-1, 1), 1)
                reconstruction_loss = lb_weighted_CEloss(predictions, label, wd)
                auc, recall, precision, correct, auc_list, fpr_dict, tpr_dict, thresh_dict = evaluate_multiclass(
                    label_onehot, pred_onehot, predictions)
            correct_label_train += correct
            auc_train += auc
            recall_train += recall
            precision_train += precision
            count += 1

            epoch_loss += reconstruction_loss.item()
            reconstruction_loss.backward()
            optimizer.step()

        test_loss, recall_test, precision_test, auc_test, correct_label_test, fpr_dict, tpr_dict, thresh_dict = \
            test_model_multiclass(model, valid_loader, data_name=data_name, cv=cv)
        train_loss_trend.append(epoch_loss / (i + 1))
        test_loss_trend.append(test_loss)

        if epoch % 1 == 0:
            print('\nEpoch %d' % epoch)
            print('Training ===>loss: ', epoch_loss / (i + 1),
                  ' Accuracy: %.2f percent' % (100 * correct_label_train / (len(train_loader.dataset))),
                  ' AUC: %.2f' % (auc_train / (i + 1)))
            print('Test ===>loss: ', test_loss,
                  ' Accuracy: %.2f percent' % (100 * correct_label_test / (len(valid_loader.dataset))),
                  ' AUC: %.2f' % auc_test)

    test_loss, recall_test, precision_test, auc_test, correct_label_test, fpr_dict, tpr_dict, thresh_dict = \
        test_model_multiclass(model, valid_loader, data_name=data_name)
    print('Test AUC: ', auc_test)

    # Save model and results
    torch.save(model.state_dict(), './ckpt/' + data_name + '/' + '_' + str(cv) + '.pt')
    ax, fig = plt.subplots()
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()

    plt.savefig(os.path.join('./plots', data_name, 'train_loss_%d.pdf' % cv))
    plt.close()

    if not os.path.exists(os.path.join("./results/", data_name)):
        os.mkdir(os.path.join("./results/", data_name))
    with open(os.path.join('./results/', data_name, 'roc_%d.pkl' % cv), 'wb') as f:
        pkl.dump({'fpr': fpr_dict, 'tpr': tpr_dict, 'thresh': thresh_dict}, f)

    return model


def train_multilabel(model, train_loader, valid_loader, optimizer, n_epochs=30, data_name='simulation',
                     cv=0):
    print('Training multi label model on ', data_name)
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    if not os.path.exists("./results/"):
        os.mkdir("./results/")
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    if not os.path.exists(os.path.join("./ckpt/", data_name)):
        os.mkdir(os.path.join("./ckpt/", data_name))
    if not os.path.exists(os.path.join("./plots/", data_name)):
        os.mkdir(os.path.join("./plots/", data_name))

    if not os.path.exists(os.path.join("./results/", data_name)):
        os.mkdir(os.path.join("./results/", data_name))

    train_loss_trend = []
    test_loss_trend = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loss_criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(n_epochs):
        model.train()
        recall_train, precision_train, auc_train, correct_label_train, epoch_loss, count = 0, 0, 0, 0, 0, 0
        i = 0
        for i, (Xx, yy) in enumerate(train_loader):
            Xd, label = Xx.to(device), yy.to(device)

            optimizer.zero_grad()
            predictions = model(Xd)
            pred_binary = predictions > 0.5

            reconstruction_loss = loss_criterion(predictions, label)
            auc, recall, precision, correct, auc_list, fpr_dict, tpr_dict, thresh_dict = evaluate_multiclass(
                label, pred_binary, predictions)

            correct_label_train += correct
            auc_train += auc
            recall_train += recall
            precision_train += precision
            count += 1

            epoch_loss += reconstruction_loss.item()
            reconstruction_loss.backward()
            optimizer.step()

        test_loss, recall_test, precision_test, auc_test, correct_label_test, fpr_dict, tpr_dict, thresh_dict = \
            test_model_multilabel(model, valid_loader, data_name=data_name, cv=cv)
        train_loss_trend.append(epoch_loss / (i + 1))
        test_loss_trend.append(test_loss)

        if epoch % 1 == 0:
            print('\nEpoch %d' % epoch)
            print('Training ===>loss: ', epoch_loss / (i + 1),
                  ' Accuracy: %.2f percent' % (100 * correct_label_train / (len(train_loader.dataset))),
                  ' AUC: %.2f' % (auc_train / (i + 1)))
            print('Test ===>loss: ', test_loss,
                  ' Accuracy: %.2f percent' % (100 * correct_label_test / (len(valid_loader.dataset))),
                  ' AUC: %.2f' % auc_test)

    test_loss, recall_test, precision_test, auc_test, correct_label_test, fpr_dict, tpr_dict, thresh_dict = \
        test_model_multilabel(model, valid_loader, data_name=data_name)
    print('Test AUC: ', auc_test)

    # Save model and results
    torch.save(model.state_dict(), './ckpt/' + data_name + '/' + '_' + str(cv) + '.pt')
    ax, fig = plt.subplots()
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plt.savefig(os.path.join('./plots', data_name, 'train_loss_%d.pdf' % cv))
    plt.close()
    with open(os.path.join('./results/', data_name, 'roc_%d.pkl' % cv), 'wb') as f:
        pkl.dump({'fpr': fpr_dict, 'tpr': tpr_dict, 'thresh': thresh_dict}, f)

    return model


def train_regression(model, train_loader, valid_loader, optimizer, n_epochs=30, data_name='simulation',
                     cv=0):
    print('Training black-box model on ', data_name)
    train_loss_trend = []
    test_loss_trend = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    if not os.path.exists("./results/"):
        os.mkdir("./results/")
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    if not os.path.exists(os.path.join("./ckpt/", data_name)):
        os.mkdir(os.path.join("./ckpt/", data_name))
    if not os.path.exists(os.path.join("./plots/", data_name)):
        os.mkdir(os.path.join("./plots/", data_name))
    if not os.path.exists(os.path.join("./results/", data_name)):
        os.mkdir(os.path.join("./results/", data_name))

    for epoch in range(n_epochs):
        model.train()
        recall_train, precision_train, auc_train, correct_label_train, epoch_loss, count = 0, 0, 0, 0, 0, 0
        i = 0
        for i, (Xx, yy) in enumerate(train_loader):
            Xd, label = Xx.to(device), yy.to(device)

            optimizer.zero_grad()
            predictions = model(Xd)

            reconstruction_loss = torch.nn.MSELoss()(predictions, label)

            epoch_loss += reconstruction_loss.item()
            reconstruction_loss.backward()
            optimizer.step()

        test_loss = 0
        for j, (Xx, yy) in enumerate(valid_loader):
            Xd, label = Xx.to(device), yy.to(device)
            predictions = model(Xd)
            test_loss += torch.nn.MSELoss()(predictions, label).item()

        train_loss_trend.append(epoch_loss / (i + 1))
        test_loss_trend.append(test_loss)

        if epoch % 1 == 0:
            print('\nEpoch %d' % epoch)
            print('Training ===>loss: ', epoch_loss / (i + 1))
            print('Test ===>loss: ', test_loss)

    test_loss = 0
    for j, (Xx, yy) in enumerate(valid_loader):
        Xd, label = Xx.to(device), yy.to(device)
        predictions = model(Xd)
        test_loss += torch.nn.MSELoss()(predictions, label).item()
    print('Test loss: ', test_loss)

    # Save model and results
    ax, fig = plt.subplots()
    torch.save(model.state_dict(), './ckpt/' + data_name + '/' + '_' + str(cv) + '.pt')
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plt.savefig(os.path.join('./plots', data_name, 'train_loss_%d.pdf' % cv))
    plt.close()

    return model


def evaluate_multiclass(labels, predicted_label, predicted_probability, data='train'):
    labels_array = labels.detach().cpu().numpy()  # one hot
    prediction_array = predicted_label.detach().cpu().numpy()  # one hot

    if len(np.unique(np.argmax(labels_array, 1))) >= 2:
        ll = np.unique(np.argmax(labels_array, 1))

        labels_array = labels_array[:, ll]

        prediction_array = prediction_array[:, ll]
        predicted_probability = predicted_probability[:, ll]

        predicted_probability = np.array(predicted_probability.detach().cpu())
        auc_list = roc_auc_score(labels_array, predicted_probability, average=None)
        auc = np.mean(auc_list)
        fpr_dict, tpr_dict, thresh_dict = {}, {}, {}

        for i, l in enumerate(np.unique(np.argmax(labels_array, 1))):
            fpr_dict[l], tpr_dict[l], thresh_dict[l] = roc_curve(labels_array[:, i], prediction_array[:, i])

        report = classification_report(labels_array, prediction_array, output_dict=True)
        recall = report['macro avg']['recall']
        precision = report['macro avg']['precision']
    else:
        auc = 0
        recall = 0
        precision = 0
        auc_list = []
        fpr_dict, tpr_dict, thresh_dict = {}, {}, {}
    correct_label = np.equal(np.argmax(labels_array, 1), np.argmax(prediction_array, 1)).sum()
    return auc, recall, precision, correct_label, auc_list, fpr_dict, tpr_dict, thresh_dict


def test_model_multiclass(model, test_loader, data_name, loss_criterion=torch.nn.CrossEntropyLoss(), cv=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    correct_label_test = 0
    recall_test, precision_test, auc_test = 0, 0, 0
    count = 0
    test_loss = 0
    auc_class_list = []
    i = 0
    fpr_dict = None
    tpr_dict = None
    thresh_dict = None
    for i, (Xx, yy, weights) in enumerate(test_loader):
        Xxd, yyd, wd = torch.Tensor(Xx.float()).to(device), torch.Tensor(yy.float()).to(device), \
            torch.Tensor(weights.float()).to(device)
        yyd = yyd.long()
        predictions = torch.nn.Softmax(dim=1)(model(Xxd))

        _, predicted_label = predictions.max(1)
        pred_onehot = torch.FloatTensor(yyd.shape[0], yyd.shape[1]).to(device)
        pred_onehot.zero_()
        pred_onehot.scatter_(1, predicted_label.view(-1, 1), 1)

        auc, recall, precision, correct, auc_list, fpr_dict, tpr_dict, thresh_dict = evaluate_multiclass(
            yyd,
            pred_onehot,
            predictions)
        auc_class_list.append(auc_list)
        correct_label_test += correct
        auc_test += auc
        recall_test += recall
        precision_test += precision
        count += 1
        _, targets = yyd.max(1)
        targets = targets.long()

        loss = loss_criterion(predictions, targets)
        test_loss += loss.item()

        for ll in range(len(fpr_dict.keys())):
            plt.figure()
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_dict[ll], tpr_dict[ll], label='Treatment: ' + str(ll))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.savefig(os.path.join('./plots', data_name, '%s_roc_%s_cv_%d.pdf' % (data_name, str(ll), cv)))
            plt.close()

    test_loss = test_loss / (i + 1)
    auc_class_list = np.array(auc_class_list).sum(0)
    print('class auc:', auc_class_list / (i + 1))

    # plot
    return test_loss, recall_test, precision_test, auc_test / (
            i + 1), correct_label_test, fpr_dict, tpr_dict, thresh_dict


def test_model_multilabel(model, test_loader, data_name, loss_criterion=torch.nn.BCEWithLogitsLoss(), cv=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    correct_label_test = 0
    recall_test, precision_test, auc_test = 0, 0, 0
    count = 0
    test_loss = 0
    auc_class_list = []
    i = 0
    fpr_dict = None
    tpr_dict = None
    thresh_dict = None
    for i, (Xx, yy) in enumerate(test_loader):
        Xxd, yyd = torch.Tensor(Xx.float()).to(device), torch.Tensor(yy.float()).to(device)

        predictions = model(Xxd)
        pred_binary = predictions > 0.5

        auc, recall, precision, correct, auc_list, fpr_dict, tpr_dict, thresh_dict = evaluate_multiclass(
            yyd,
            pred_binary,
            predictions)
        auc_class_list.append(auc_list)
        correct_label_test += correct
        auc_test += auc
        recall_test += recall
        precision_test += precision
        count += 1

        loss = loss_criterion(predictions, yyd)
        test_loss += loss.item()

        for ll in range(len(fpr_dict.keys())):
            plt.figure()
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_dict[ll], tpr_dict[ll], label='Treatment: ' + str(ll))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.savefig(os.path.join('./plots', data_name, '%s_roc_%d_cv_%d.pdf' % (data_name, ll, cv)))

    test_loss = test_loss / (i + 1)
    auc_class_list = np.array(auc_class_list).sum(0)
    print('class auc:', auc_class_list / (i + 1))

    # plot
    return test_loss, recall_test, precision_test, auc_test / (
            i + 1), correct_label_test, fpr_dict, tpr_dict, thresh_dict


def test_model_weightedCE(model, test_loader, data_name, n, d=2, cv=0, n_targets=2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    correct_label_test = 0
    recall_test, precision_test, auc_test = 0, 0, 0
    count = 0
    test_loss = 0
    auc_class_list = []
    fpr_dict, tpr_dict, thresh_dict = None, None, None
    i = 0
    for i, (Xx, yy, weights) in enumerate(test_loader):
        Xxd, yyd, wd = torch.Tensor(Xx.float()).to(device), torch.Tensor(yy.float()).to(device), \
            torch.Tensor(weights.float()).to(device)

        yyd = yyd.long()
        predictions = torch.nn.Softmax(dim=1)(model(Xxd))

        label_onehot = torch.FloatTensor(yyd.size(0), n_targets).to(device)
        label_onehot.zero_()
        label_onehot.scatter_(1, yyd.view(-1, 1), 1)

        _, predicted_label = predictions.max(1)
        pred_onehot = torch.FloatTensor(yyd.size(0), n_targets).to(device)
        pred_onehot.zero_()
        pred_onehot.scatter_(1, predicted_label.view(-1, 1), 1)

        if data_name != 'Ours':
            auc, recall, precision, correct, auc_list, fpr_dict, tpr_dict, thresh_dict = evaluate_multiclass(
                label_onehot,
                pred_onehot,
                predictions, data='test')
        else:
            auc, recall, precision, correct, auc_list, fpr_dict, tpr_dict, thresh_dict = evaluate_multiclass(
                label_onehot,
                pred_onehot,
                predictions, data='test')
        auc_class_list.append(auc_list)
        correct_label_test += correct
        auc_test += auc
        recall_test += recall
        precision_test += precision
        count += 1

        loss = lb_weightedCE(predictions, wd)
        test_loss += loss.item()

        for ll in range(len(fpr_dict.keys())):
            plt.figure()
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_dict[ll], tpr_dict[ll], label='Treatment: ' + str(ll))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.savefig(os.path.join('./plots', data_name,
                                     'roc_label_%d_%s_n_%d_d_%d_cv_%d.pdf' % (ll, 'weighted_ce_loss', n, d, cv)))

    test_loss = test_loss / (i + 1)
    auc_class_list = np.array(auc_class_list).sum(0)
    print('class auc:', auc_class_list / (i + 1))

    # plot
    return test_loss, recall_test, precision_test, auc_test / (
            i + 1), correct_label_test, fpr_dict, tpr_dict, thresh_dict


def eval_via_rollout(data_obj, n, data_name='synthetic_continuous', d=2, maxcv=0, n_tests=100,
                     loss_type='weighted_cross_entropy', train_policies=False):
    if data_name == 'synthetic_continuous' or data_name == 'synthetic_no_bounds':
        c_dim = data_obj.c_dim

        for cv in range(maxcv):
            model_obs_expc1_expc2 = NNmodel(input_size=c_dim, hidden_size=HIDDEN_SIZE, target_size=2)
            if loss_type == 'smooth_max':
                model_obs_expc1_expc2.load_state_dict(
                    torch.load(os.path.join('./ckpt', data_name,
                                            'obs_expc1_expc2_%s_n_%d_d_%d_cv_%d.pt' % (loss_type, n, d, cv))))
            else:
                model_obs_expc1_expc2.load_state_dict(
                    torch.load(os.path.join('./ckpt', data_name, 'obs_expc1_expc2_n_%d_d_%d_cv_%d.pt' % (n, d, cv))))
            model_obs_expc1_expc2.eval()

            model_obs = NNmodel(input_size=c_dim, hidden_size=HIDDEN_SIZE, target_size=2)
            if loss_type == 'smooth_max':
                model_obs.load_state_dict(
                    torch.load(os.path.join('./ckpt', data_name, 'obs_%s_n_%d_d_%d_cv_%d.pt' % (loss_type, n, d, cv))))
            else:
                model_obs.load_state_dict(
                    torch.load(os.path.join('./ckpt', data_name, 'obs_n_%d_d_%d_cv_%d.pt' % (n, d, cv))))
            model_obs.eval()

            model_obs_expc1 = NNmodel(input_size=c_dim, hidden_size=HIDDEN_SIZE, target_size=2)
            if loss_type == 'smooth_max':
                model_obs_expc1.load_state_dict(
                    torch.load(
                        os.path.join('./ckpt', data_name, 'obs_expc1_%s_n_%d_d_%d_cv_%d.pt' % (loss_type, n, d, cv))))
            else:
                model_obs_expc1.load_state_dict(
                    torch.load(os.path.join('./ckpt', data_name, 'obs_expc1_n_%d_d_%d_cv_%d.pt' % (n, d, cv))))
            model_obs_expc1.eval()

            model_obs_expc2 = NNmodel(input_size=c_dim, hidden_size=HIDDEN_SIZE, target_size=2)
            if loss_type == 'smooth_max':
                model_obs_expc2.load_state_dict(
                    torch.load(
                        os.path.join('./ckpt', data_name, 'obs_expc2_%s_n_%d_d_%d_cv_%d.pt' % (loss_type, n, d, cv))))
            else:
                model_obs_expc2.load_state_dict(
                    torch.load(os.path.join('./ckpt', data_name, 'obs_expc2_n_%d_d_%d_cv_%d.pt' % (n, d, cv))))
            model_obs_expc2.eval()

            def random_policy():
                return np.random.uniform(0, 1, size=1)[0]

            # np.random.seed(30)  # generate the same data always, randomness only in the models here - this is a random
            # test set!
            u_mat = np.random.normal(loc=data_obj.u_mean, scale=data_obj.u_cov, size=(n_tests, data_obj.u_dim))

            # sample c - hard coded u dimension here - beware!
            c_test = np.zeros((n_tests, c_dim))
            for dd in range(c_dim):
                c_test[:, dd] = data_obj.alpha_c[dd] * u_mat[:, 0] + \
                                data_obj.beta_c[dd] + 1 * np.random.normal(0, 1, size=n_tests)

            # sample x, y
            p_x_behavior = expit(
                np.dot(np.concatenate((u_mat, c_test), axis=1), data_obj.alpha_x) + np.random.normal(0, scale=1,
                                                                                                     size=n_tests) - data_obj.x_th)
            x_behavior = np.double(p_x_behavior > data_obj.x_th)

            if train_policies:
                train_wrapper_bounds(X=c_test, targets=x_behavior, lower_bounds=None, data_name='synthetic_pxgivenc1c2',
                                     loss_type='cross_entropy', n_targets=2, n_epochs=30,
                                     hidden_size=HIDDEN_SIZE, model='XGBoost', cv=cv)
                train_wrapper_bounds(X=c_test[:, 0].reshape(-1, 1), targets=x_behavior, lower_bounds=None,
                                     data_name='synthetic_pxgivenc1', loss_type='cross_entropy', n_targets=2,
                                     n_epochs=30, hidden_size=HIDDEN_SIZE, model='XGBoost', cv=cv)
                train_wrapper_bounds(X=c_test[:, 1].reshape(-1, 1), targets=x_behavior, lower_bounds=None,
                                     data_name='synthetic_pxgivenc2', loss_type='cross_entropy', n_targets=2,
                                     n_epochs=30, hidden_size=HIDDEN_SIZE, model='XGBoost', cv=cv)
            model_pxgivenc1c2 = XGBModel(input_size=c_test.shape[1],
                                         target_size=2)

            t = 1000 * time.time()  # current time in milliseconds
            np.random.seed(int(t) % 2 ** 32)  # reset seed when nothing to do with data
            x_random = np.random.binomial(1, 0.5, size=n_tests)

            p_x_random = np.random.uniform(0, 1, n_tests)
            p_x_model_obs_expc1_expc2 = torch.nn.Softmax(dim=1)(
                model_obs_expc1_expc2(torch.Tensor(c_test).float())).detach().numpy()
            p_x_model_obs_expc1 = torch.nn.Softmax(dim=1)(
                model_obs_expc1(torch.Tensor(c_test).float())).detach().numpy()
            p_x_model_obs_expc2 = torch.nn.Softmax(dim=1)(
                model_obs_expc2(torch.Tensor(c_test).float())).detach().numpy()
            p_x_model_obs = torch.nn.Softmax(dim=1)(model_obs(torch.Tensor(c_test).float())).detach().numpy()
            p_x_oracle = np.ones(n_tests)

            thr_sweep = np.linspace(0.1, 1, 20)
            mean_y_model_obs_expc1_expc2 = np.zeros(len(thr_sweep))
            mean_y_model_obs_expc1 = np.zeros(len(thr_sweep))
            mean_y_model_obs_expc2 = np.zeros(len(thr_sweep))
            mean_y_model_obs = np.zeros(len(thr_sweep))
            mean_y_behavior = np.zeros(len(thr_sweep))
            mean_y_random = np.zeros(len(thr_sweep))
            mean_y_oracle = np.zeros(len(thr_sweep))

            mean_y_model_obs_expc1_expc2_select_top = np.zeros(len(thr_sweep))
            mean_y_model_obs_expc1_select_top = np.zeros(len(thr_sweep))
            mean_y_model_obs_expc2_select_top = np.zeros(len(thr_sweep))
            mean_y_model_obs_select_top = np.zeros(len(thr_sweep))
            mean_y_behavior_select_top = np.zeros(len(thr_sweep))
            mean_y_random_select_top = np.zeros(len(thr_sweep))
            mean_y_oracle_select_top = np.zeros(len(thr_sweep))

            rollout_df = pd.DataFrame(
                columns=['th', 'pc_model', 'pc_random', 'pc_behavior',
                         'n', 'x_model', 'x_behavior', 'x_oracle',
                         'x_random', 'y_model', 'y_behavior',
                         'y_random', 'y_oracle', 'mean_y_model_obs_expc1_expc2', 'mean_y_model_obs_expc1',
                         'mean_y_model_obs_expc2',
                         'mean_y_model_obs',
                         'mean_y_behavior',
                         'mean_y_model_obs_expc1_expc2_select_top',
                         'mean_y_model_obs_expc1_select_top',
                         'mean_y_model_obs_expc2_select_top',
                         'mean_y_model_obs_select_top',
                         'mean_y_behavior_select_top',
                         'mean_y_random_select_top',
                         'mean_y_oracle_select_top',
                         'mean_y_model_obs_expc1_expc2_ipw',
                         'mean_y_model_obs_expc1_ipw',
                         'mean_y_model_obs_expc2_ipw',
                         'mean_y_model_obs_ipw',
                         'mean_y_behavior_ipw',
                         'mean_y_random_ipw',
                         'mean_y_oracle_ipw'
                         'cv'])

            for th_count, th in enumerate(thr_sweep):
                x_model_obs_expc1_expc2 = np.asarray(
                    [p_x_model_obs_expc1_expc2[i, 1] >= th for i in range(n_tests)]).astype(float)
                x_model_obs_expc1 = np.asarray([p_x_model_obs_expc1[i, 1] >= th for i in range(n_tests)]).astype(float)
                x_model_obs_expc2 = np.asarray([p_x_model_obs_expc2[i, 1] >= th for i in range(n_tests)]).astype(float)
                x_model_obs = np.asarray([p_x_model_obs[i, 1] >= th for i in range(n_tests)]).astype(float)

                if data_name == "synthetic_continuous":
                    x_oracle = np.ones(n_tests)
                    # sample y
                    y_behavior = np.double(
                        expit(np.dot(np.concatenate((u_mat, c_test, x_behavior.reshape(-1, 1)), axis=1),
                                     data_obj.alpha_y) - data_obj.y_th) > 0.5)

                    y_model_obs_expc1_expc2 = (np.dot(data_obj.alpha_y,
                                                      np.concatenate(
                                                          (u_mat, c_test, x_model_obs_expc1_expc2.reshape(-1, 1)),
                                                          axis=1).T) - data_obj.y_th > 0).flatten().astype(float)

                    y_model_obs_expc1 = (np.dot(data_obj.alpha_y,
                                                np.concatenate(
                                                    (u_mat, c_test, x_model_obs_expc1.reshape(-1, 1)),
                                                    axis=1).T) - data_obj.y_th > 0).flatten().astype(float)

                    y_model_obs_expc2 = (np.dot(data_obj.alpha_y,
                                                np.concatenate(
                                                    (u_mat, c_test, x_model_obs_expc2.reshape(-1, 1)),
                                                    axis=1).T) - data_obj.y_th > 0).flatten().astype(float)

                    y_model_obs = (np.dot(data_obj.alpha_y,
                                          np.concatenate(
                                              (u_mat, c_test, x_model_obs.reshape(-1, 1)),
                                              axis=1).T) - data_obj.y_th > 0).flatten().astype(float)

                    y_random = (np.dot(data_obj.alpha_y, np.concatenate((u_mat, c_test, x_random.reshape(-1, 1)),
                                                                        axis=1).T) + data_obj.y_th > 0).flatten().astype(
                        float)
                    y_oracle = (np.dot(data_obj.alpha_y, np.concatenate((u_mat, c_test, x_oracle.reshape(-1, 1)),
                                                                        axis=1).T) + data_obj.y_th > 0).flatten().astype(
                        float)
                elif data_name == "synthetic_no_bounds":
                    # sample y
                    x_oracle = np.ones(n_tests)
                    y_behavior = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_behavior)
                    y_model_obs_expc1_expc2 = data_obj.sample_y(u_vec=u_mat, c_vec=c_test,
                                                                x_vec=x_model_obs_expc1_expc2)
                    y_model_obs_expc1 = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_model_obs_expc1)
                    y_model_obs_expc2 = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_model_obs_expc2)
                    y_model_obs = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_model_obs)
                    y_random = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_random)
                    y_oracle = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_oracle)

                mean_y_model_obs_expc1_expc2[th_count] = np.mean(y_model_obs_expc1_expc2)
                mean_y_model_obs_expc1[th_count] = np.mean(y_model_obs_expc1)
                mean_y_model_obs_expc2[th_count] = np.mean(y_model_obs_expc2)
                mean_y_model_obs[th_count] = np.mean(y_model_obs)
                mean_y_random[th_count] = np.mean(y_random)
                mean_y_behavior[th_count] = np.mean(y_behavior)
                mean_y_oracle[th_count] = np.mean(y_oracle)

                x_model_obs_expc1_expc2_st = np.zeros(n_tests)
                x_model_obs_expc1_expc2_st[(np.argsort(p_x_model_obs_expc1_expc2[:, 1])[::-1])[:int(n_tests * th)]] = 1

                x_model_obs_expc1_st = np.zeros(n_tests)
                x_model_obs_expc1_st[(np.argsort(p_x_model_obs_expc1[:, 1])[::-1])[:int(n_tests * th)]] = 1

                x_model_obs_expc2_st = np.zeros(n_tests)
                x_model_obs_expc2_st[(np.argsort(p_x_model_obs_expc2[:, 1])[::-1])[:int(n_tests * th)]] = 1

                x_model_obs_st = np.zeros(n_tests)
                x_model_obs_st[(np.argsort(p_x_model_obs[:, 1])[::-1])[:int(n_tests * th)]] = 1

                x_random_st = np.zeros(n_tests)
                x_random_st[np.argsort(p_x_random)[:int(n_tests * th)]] = 1

                x_behavior_st = np.zeros(n_tests)
                x_behavior_st[np.argsort(p_x_behavior)[:int(n_tests * th)]] = 1

                x_oracle_st = np.zeros(n_tests)
                x_oracle_st[np.argsort(p_x_oracle)[:int(n_tests * th)]] = 1

                # sample y
                if data_name == "synthetic_continuous":
                    y_behavior = np.double(
                        expit(np.dot(np.concatenate((u_mat, c_test, x_behavior_st.reshape(-1, 1)), axis=1),
                                     data_obj.alpha_y) - data_obj.y_th) > 0.5)

                    y_model_obs_expc1_expc2 = (np.dot(data_obj.alpha_y,
                                                      np.concatenate(
                                                          (u_mat, c_test, x_model_obs_expc1_expc2_st.reshape(-1, 1)),
                                                          axis=1).T) - data_obj.y_th > 0).flatten().astype(float)

                    y_model_obs_expc1 = (np.dot(data_obj.alpha_y,
                                                np.concatenate(
                                                    (u_mat, c_test, x_model_obs_expc1_st.reshape(-1, 1)),
                                                    axis=1).T) - data_obj.y_th > 0).flatten().astype(float)

                    y_model_obs_expc2 = (np.dot(data_obj.alpha_y,
                                                np.concatenate(
                                                    (u_mat, c_test, x_model_obs_expc2_st.reshape(-1, 1)),
                                                    axis=1).T) - data_obj.y_th > 0).flatten().astype(float)

                    y_model_obs = (np.dot(data_obj.alpha_y,
                                          np.concatenate(
                                              (u_mat, c_test, x_model_obs_st.reshape(-1, 1)),
                                              axis=1).T) - data_obj.y_th > 0).flatten().astype(float)

                    y_random = (np.dot(data_obj.alpha_y, np.concatenate((u_mat, c_test, x_random_st.reshape(-1, 1)),
                                                                        axis=1).T) + data_obj.y_th > 0).flatten().astype(
                        float)

                    y_oracle = (np.dot(data_obj.alpha_y, np.concatenate((u_mat, c_test, x_oracle_st.reshape(-1, 1)),
                                                                        axis=1).T) + data_obj.y_th > 0).flatten().astype(
                        float)
                elif data_name == "synthetic_no_bounds":
                    y_behavior = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_behavior_st)
                    y_model_obs_expc1_expc2 = data_obj.sample_y(u_vec=u_mat, c_vec=c_test,
                                                                x_vec=x_model_obs_expc1_expc2_st)
                    y_model_obs_expc1 = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_model_obs_expc1_st)
                    y_model_obs_expc2 = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_model_obs_expc2_st)
                    y_model_obs = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_model_obs_st)
                    y_random = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_random_st)
                    y_oracle = data_obj.sample_y(u_vec=u_mat, c_vec=c_test, x_vec=x_oracle_st)

                mean_y_model_obs_expc1_expc2_select_top[th_count] = np.mean(y_model_obs_expc1_expc2)
                mean_y_model_obs_expc1_select_top[th_count] = np.mean(y_model_obs_expc1)
                mean_y_model_obs_expc2_select_top[th_count] = np.mean(y_model_obs_expc2)
                mean_y_model_obs_select_top[th_count] = np.mean(y_model_obs)
                mean_y_random_select_top[th_count] = np.mean(y_random)
                mean_y_behavior_select_top[th_count] = np.mean(y_behavior)

                # ipw
                print(p_x_behavior.shape, p_x_model_obs_expc1_expc2.shape)
                mean_y_model_obs_expc1_expc2_ipw = np.mean(
                    (p_x_model_obs_expc1_expc2[:, 1] / p_x_behavior) * y_behavior * x_model_obs_expc1_expc2
                    + (1 - p_x_model_obs_expc1_expc2[:, 1]) / (1 - p_x_behavior) * (
                            1 - x_model_obs_expc1_expc2) * y_behavior)
                mean_y_model_obs_expc1_ipw = np.mean(
                    (p_x_model_obs_expc1[:, 1] / p_x_behavior) * y_behavior * x_model_obs_expc1
                    + (1 - p_x_model_obs_expc1[:, 1]) / (1 - p_x_behavior) * (1 - x_model_obs_expc1) * y_behavior)
                mean_y_model_obs_expc2_ipw = np.mean(
                    (p_x_model_obs_expc2[:, 1] / p_x_behavior) * y_behavior * x_model_obs_expc2
                    + (1 - p_x_model_obs_expc2[:, 1]) / (1 - p_x_behavior) * (1 - x_model_obs_expc2) * y_behavior)
                mean_y_model_obs_ipw = np.mean(
                    (p_x_model_obs[:, 1] / p_x_behavior) * y_behavior * x_model_obs
                    + (1 - p_x_model_obs[:, 1]) / (1 - p_x_behavior) * (1 - x_model_obs) * y_behavior)
                mean_y_behavior_ipw = np.mean(
                    (p_x_behavior / p_x_behavior) * y_behavior * x_behavior
                    + (1 - p_x_behavior) / (1 - p_x_behavior) * (1 - x_behavior) * y_behavior)
                mean_y_random_ipw = np.mean(
                    (p_x_random / p_x_behavior) * y_behavior * x_random
                    + (1 - p_x_random) / (1 - p_x_behavior) * (1 - x_random) * y_behavior)
                mean_y_oracle_ipw = np.mean(
                    (p_x_oracle / p_x_behavior) * y_behavior * x_oracle
                    + (1 - p_x_oracle) / (1 - p_x_behavior) * (1 - x_oracle) * y_behavior)

                rollout_df = rollout_df.append({'th': th,
                                                'n': np.arange(n_tests),
                                                'pc_model': len(np.where(x_model_obs_expc1_expc2 == 1)[0]) / n_tests,
                                                'pc_random': len(np.where(x_random == 1)[0]) / n_tests,
                                                'pc_behavior': len(np.where(x_behavior == 1)[0]) / n_tests,
                                                'x_model_obs_expc1_expc2': x_model_obs_expc1_expc2,
                                                'x_model_obs_expc1': x_model_obs_expc1,
                                                'x_model_obs_expc2': x_model_obs_expc2,
                                                'x_model_obs': x_model_obs,
                                                'x_behavior': x_behavior, 'x_random': x_random,
                                                'x_oracle': x_oracle,
                                                'y_model_obs_expc1_expc2': y_model_obs_expc1_expc2,
                                                'y_model_obs_expc1': y_model_obs_expc1,
                                                'y_model_obs_expc2': y_model_obs_expc2,
                                                'y_model_obs': y_model_obs,
                                                'y_random': y_random,
                                                'y_behavior': y_behavior,
                                                'y_oracle': y_oracle,
                                                'mean_y_model_obs_expc1_expc2': mean_y_model_obs_expc1_expc2[th_count],
                                                'mean_y_model_obs_expc1': mean_y_model_obs_expc1[th_count],
                                                'mean_y_model_obs_expc2': mean_y_model_obs_expc2[th_count],
                                                'mean_y_model_obs': mean_y_model_obs[th_count],
                                                'mean_y_behavior': mean_y_behavior[th_count],
                                                'mean_y_random': mean_y_random[th_count],
                                                'mean_y_oracle': mean_y_oracle[th_count],
                                                'mean_y_model_obs_expc1_expc2_select_top':
                                                    mean_y_model_obs_expc1_expc2_select_top[th_count],
                                                'mean_y_model_obs_expc1_select_top': mean_y_model_obs_expc1_select_top[
                                                    th_count],
                                                'mean_y_model_obs_expc2_select_top': mean_y_model_obs_expc2_select_top[
                                                    th_count],
                                                'mean_y_model_obs_select_top': mean_y_model_obs_select_top[th_count],
                                                'mean_y_behavior_select_top': mean_y_behavior_select_top[th_count],
                                                'mean_y_random_select_top': mean_y_random_select_top[th_count],
                                                'mean_y_oracle_select_top': mean_y_oracle_select_top[th_count],
                                                'mean_y_model_obs_expc1_expc2_ipw': mean_y_model_obs_expc1_expc2_ipw,
                                                'mean_y_model_obs_expc1_ipw': mean_y_model_obs_expc1_ipw,
                                                'mean_y_model_obs_expc2_ipw': mean_y_model_obs_expc2_ipw,
                                                'mean_y_model_obs_ipw': mean_y_model_obs_ipw,
                                                'mean_y_behavior_ipw': mean_y_behavior_ipw,
                                                'mean_y_random_ipw': mean_y_random_ipw,
                                                'mean_y_oracle_ipw': mean_y_oracle_ipw,
                                                'cv': cv},
                                               ignore_index=True)
                print(
                    "th: %.3f, mean_y_model: %.3f, mean_y_behavior: %.3f, cv: %d" % (th, mean_y_model_obs_expc1_expc2[th_count],
                                                                             mean_y_behavior[th_count], cv))
        plot_rollout_eval(rollout_df, data_name=data_name, maxcv=maxcv)
        rollout_df.to_pickle(os.path.join('./results', data_name, 'rollout_df_%d.pkl' % maxcv))
    else:
        ValueError('data_name not recognized')


def eval_via_pseudo_rollout(data_dict, data_name, n_targets=2, maxcv=1):
    """
    Evaluate the model via a pseudo rollout
    :param data_dict: dictionary containing the data
    :param data_name: name of the dataset
    :param n_targets: number of targets
    :param maxcv: max cross validation folds
    dumps all results to file
    """
    if data_name == 'IST':
        c1_dims = data_dict['c1_dims']
        c2_dims = data_dict['c2_dims']
        C_test_c1 = np.asarray(data_dict['test_df_expc1'][c1_dims])
        C_test_c2 = np.asarray(data_dict['test_df_expc2'][c2_dims])
        C_test = np.asarray(np.concatenate((C_test_c1, C_test_c2), axis=1))
        n_samples_train = data_dict['weight_matrix_train_0'].shape[0]
        n_samples_test = data_dict['test_df_obs'].shape[0]
        test_df_expc1 = data_dict['test_df_expc1']
        x_test_exp = np.asarray(test_df_expc1['X'])
        y_test_exp = np.asarray(test_df_expc1['Outcome'])
        C_train_obs = np.asarray(data_dict['train_df_obs'][c1_dims + c2_dims])
        y_train_obs = np.asarray(data_dict['train_df_obs']['Outcome'])

        rollout_df = pd.DataFrame(
            columns=['th', 'pc_model', 'pc_behavior',
                     'n', 'x_model', 'x_behavior',
                     'y_model',
                     'y_behavior',
                     'mean_y_model', 'mean_y_behavior', 'cv'])

        for cv in range(maxcv):
            # learned model
            print('reading model from path:', os.path.join('./ckpt', data_name,
                                                           '%s_n_%d_d_%d_cv_%d.pt' % ('obs_expc1_expc2',
                                                                                      n_samples_train + n_samples_test,
                                                                                      C_test.shape[1],
                                                                                      cv)))
            model_obs_expc1_expc2 = NNmodel(input_size=C_test.shape[1], hidden_size=HIDDEN_SIZE, target_size=n_targets)
            model_obs_expc1_expc2.load_state_dict(torch.load(os.path.join('./ckpt', data_name,
                                                                          '%s_n_%d_d_%d_cv_%d.pt' % ('obs_expc1_expc2',
                                                                                                     n_samples_train + n_samples_test,
                                                                                                     C_test.shape[1],
                                                                                                     cv))))

            model_obs_expc1 = NNmodel(input_size=C_test.shape[1], hidden_size=HIDDEN_SIZE, target_size=n_targets)
            model_obs_expc1.load_state_dict(torch.load(os.path.join('./ckpt', data_name,
                                                                    '%s_n_%d_d_%d_cv_%d.pt' % ('obs_expc1',
                                                                                               n_samples_train + n_samples_test,
                                                                                               C_test.shape[1],
                                                                                               cv))))

            model_obs_expc2 = NNmodel(input_size=C_test.shape[1], hidden_size=HIDDEN_SIZE, target_size=n_targets)
            model_obs_expc2.load_state_dict(torch.load(os.path.join('./ckpt', data_name,
                                                                    '%s_n_%d_d_%d_cv_%d.pt' % ('obs_expc2',
                                                                                               n_samples_train + n_samples_test,
                                                                                               C_test.shape[1],
                                                                                               cv))))

            model_obs = NNmodel(input_size=C_test.shape[1], hidden_size=HIDDEN_SIZE, target_size=n_targets)
            model_obs.load_state_dict(torch.load(os.path.join('./ckpt', data_name,
                                                              '%s_n_%d_d_%d_cv_%d.pt' % ('obs',
                                                                                         n_samples_train + n_samples_test,
                                                                                         C_test.shape[1],
                                                                                         cv))))

            # these two are for ipw
            model_behavior = XGBModel(input_size=len(c1_dims) + len(c2_dims), target_size=len(np.unique(x_test_exp)))
            model_behavior.load(
                os.path.join('./ckpt', data_name + '_bounds' + '_px_givenc1c2' + '_bounds',
                             'xXGBoostModel_%d.pkl' % cv))

            print('reading outcome model from path:',
                  os.path.join('./ckpt', 'IST_bounds_py_givenxc1c2' + '_bounds', 'xXGBoostModel_%d.pkl' % cv))
            model_py_givenxc1c2 = XGBModel(input_size=len(c1_dims) + len(c2_dims) + 1, target_size=2)
            model_py_givenxc1c2.load(
                os.path.join('./ckpt', 'IST_bounds_py_givenxc1c2' + '_bounds', 'xXGBoostModel_%d.pkl' % cv))

            p_x_model_obs_expc1_expc2 = torch.nn.Softmax(dim=1)(
                model_obs_expc1_expc2(torch.Tensor(C_test))).float().detach().numpy()[:, 1]
            p_x_model_obs_expc1 = torch.nn.Softmax(dim=1)(
                model_obs_expc1(torch.Tensor(C_test))).float().detach().numpy()[:,
                                  1]
            p_x_model_obs_expc2 = torch.nn.Softmax(dim=1)(
                model_obs_expc2(torch.Tensor(C_test))).float().detach().numpy()[:,
                                  1]
            p_x_model_obs = torch.nn.Softmax(dim=1)(model_obs(torch.Tensor(C_test))).float().detach().numpy()[:, 1]

            # ipw
            p_x_behavior_train = model_behavior.predict(C_train_obs)[:, 1]
            # print(p_x_behavior_train)
            # exit(1)
            p_x_model_obs_expc1_expc2_train = torch.nn.Softmax(dim=1)(
                model_obs_expc1_expc2(torch.Tensor(C_train_obs))).float().detach().numpy()[:, 1]
            p_x_model_obs_expc1_train = torch.nn.Softmax(dim=1)(
                model_obs_expc1(torch.Tensor(C_train_obs))).float().detach().numpy()[:,
                                        1]
            p_x_model_obs_expc2_train = torch.nn.Softmax(dim=1)(
                model_obs_expc2(torch.Tensor(C_train_obs))).float().detach().numpy()[:,
                                        1]
            p_x_model_obs_train = torch.nn.Softmax(dim=1)(
                model_obs(torch.Tensor(C_train_obs))).float().detach().numpy()[:, 1]

            thr_sweep = np.linspace(0, 1, 20)
            mean_y_model_obs_expc1_expc2 = np.zeros(len(thr_sweep))
            mean_y_model_obs_expc1 = np.zeros(len(thr_sweep))
            mean_y_model_obs_expc2 = np.zeros(len(thr_sweep))
            mean_y_model_obs = np.zeros(len(thr_sweep))
            mean_y_behavior = np.zeros(len(thr_sweep))

            mean_y_model_obs_expc1_expc2_select_top = np.zeros(len(thr_sweep))
            mean_y_model_obs_expc1_select_top = np.zeros(len(thr_sweep))
            mean_y_model_obs_expc2_select_top = np.zeros(len(thr_sweep))
            mean_y_model_obs_select_top = np.zeros(len(thr_sweep))
            mean_y_behavior_select_top = np.zeros(len(thr_sweep))

            n_tests = C_test.shape[0]
            for th_count, th in enumerate(thr_sweep):

                x_behavior = np.asarray(data_dict['test_df_obs']['X']).astype(float)
                x_behavior_obs_train = np.asarray(data_dict['train_df_obs']['X']).astype(float)

                x_model_obs_expc1_expc2 = np.asarray(
                    [p_x_model_obs_expc1_expc2[i] >= th for i in range(n_tests)]).astype(float)

                x_model_obs_expc1 = np.asarray([p_x_model_obs_expc1[i] >= th for i in range(n_tests)]).astype(float)

                x_model_obs_expc2 = np.asarray([p_x_model_obs_expc2[i] >= th for i in range(n_tests)]).astype(float)

                x_model_obs = np.asarray([p_x_model_obs[i] >= th for i in range(n_tests)]).astype(float)

                y_behavior = np.asarray(
                    [y_test_exp[i] for i in range(n_tests) if test_df_expc1['Z'].iloc[i] == 1]).astype(
                    float)

                y_behavior_obs_train = y_train_obs

                y_model_obs_expc1_expc2 = np.asarray(
                    [y_test_exp[i] for i in range(n_tests) if x_test_exp[i] == x_model_obs_expc1_expc2[i]]).astype(
                    float)

                y_model_obs_expc1 = np.asarray(
                    [y_test_exp[i] for i in range(n_tests) if x_test_exp[i] == x_model_obs_expc1[i]]).astype(float)
                y_model_obs_expc2 = np.asarray(
                    [y_test_exp[i] for i in range(n_tests) if x_test_exp[i] == x_model_obs_expc2[i]]).astype(float)
                y_model_obs = np.asarray(
                    [y_test_exp[i] for i in range(n_tests) if x_test_exp[i] == x_model_obs[i]]).astype(float)

                mean_y_model_obs_expc1_expc2[th_count] = np.mean(y_model_obs_expc1_expc2)
                mean_y_model_obs_expc1[th_count] = np.mean(y_model_obs_expc1)
                mean_y_model_obs_expc2[th_count] = np.mean(y_model_obs_expc2)
                mean_y_model_obs[th_count] = np.mean(y_model_obs)
                mean_y_behavior[th_count] = np.mean(y_behavior)

                x_model_obs_expc1_expc2_st = np.zeros(n_tests)
                x_model_obs_expc1_expc2_st[(np.argsort(p_x_model_obs_expc1_expc2)[::-1])[:int(n_tests * th)]] = 1

                x_model_obs_expc1_st = np.zeros(n_tests)
                x_model_obs_expc1_st[(np.argsort(p_x_model_obs_expc1)[::-1])[:int(n_tests * th)]] = 1

                x_model_obs_expc2_st = np.zeros(n_tests)
                x_model_obs_expc2_st[(np.argsort(p_x_model_obs_expc2)[::-1])[:int(n_tests * th)]] = 1

                x_model_obs_st = np.zeros(n_tests)
                x_model_obs_st[(np.argsort(p_x_model_obs)[::-1])[:int(n_tests * th)]] = 1

                # sample y
                y_all_behavior = np.asarray([y_test_exp[i] for i in range(n_tests) if test_df_expc1['Z'].iloc[i] == 1])
                if int(n_tests * th) >= len(y_all_behavior):
                    y_behavior = y_all_behavior
                else:
                    y_behavior = np.random.choice(y_all_behavior, int(n_tests * th), replace=False)

                y_model_obs_expc1_expc2 = np.asarray(
                    [y_test_exp[i] for i in range(n_tests) if x_test_exp[i] == x_model_obs_expc1_expc2_st[i]]).astype(
                    float)

                y_model_obs_expc1 = np.asarray(
                    [y_test_exp[i] for i in range(n_tests) if x_test_exp[i] == x_model_obs_expc1_st[i]]).astype(float)
                y_model_obs_expc2 = np.asarray(
                    [y_test_exp[i] for i in range(n_tests) if x_test_exp[i] == x_model_obs_expc2_st[i]]).astype(float)
                y_model_obs = np.asarray(
                    [y_test_exp[i] for i in range(n_tests) if x_test_exp[i] == x_model_obs_st[i]]).astype(float)

                mean_y_model_obs_expc1_expc2_select_top[th_count] = np.mean(y_model_obs_expc1_expc2)
                mean_y_model_obs_expc1_select_top[th_count] = np.mean(y_model_obs_expc1)
                mean_y_model_obs_expc2_select_top[th_count] = np.mean(y_model_obs_expc2)
                mean_y_model_obs_select_top[th_count] = np.mean(y_model_obs)
                mean_y_behavior_select_top[th_count] = np.mean(y_behavior)

                print(p_x_behavior_train.shape, p_x_model_obs_expc1_expc2_train.shape, x_behavior_obs_train.shape,y_behavior_obs_train.shape)
                mean_y_model_obs_expc1_expc2_ipw = np.mean(
                    (p_x_model_obs_expc1_expc2_train / p_x_behavior_train) * y_behavior_obs_train * x_behavior_obs_train
                    + (1 - p_x_model_obs_expc1_expc2_train) / (1 - p_x_behavior_train) * (
                            1 - x_behavior_obs_train) * y_behavior_obs_train)
                mean_y_model_obs_expc1_ipw = np.mean(
                    (p_x_model_obs_expc1_train / p_x_behavior_train) * y_behavior_obs_train * x_behavior_obs_train
                    + (1 - p_x_model_obs_expc1_train) / (1 - p_x_behavior_train) * (
                            1 - x_behavior_obs_train) * y_behavior_obs_train)
                mean_y_model_obs_expc2_ipw = np.mean(
                    (p_x_model_obs_expc2_train / p_x_behavior_train) * y_behavior_obs_train * x_behavior_obs_train
                    + (1 - p_x_model_obs_expc2_train) / (1 - p_x_behavior_train) * (
                            1 - x_behavior_obs_train) * y_behavior_obs_train)
                mean_y_model_obs_ipw = np.mean(
                    (p_x_model_obs_train / p_x_behavior_train) * y_behavior_obs_train * x_behavior_obs_train
                    + (1 - p_x_model_obs_train) / (1 - p_x_behavior_train) * (
                            1 - x_behavior_obs_train) * y_behavior_obs_train)
                mean_y_behavior_ipw = np.mean(
                    (p_x_behavior_train / p_x_behavior_train) * y_behavior_obs_train * x_behavior_obs_train
                    + (1 - p_x_behavior_train) / (1 - p_x_behavior_train) * (
                            1 - x_behavior_obs_train) * y_behavior_obs_train)

                rollout_df = rollout_df.append({'th': th,
                                                'n': np.arange(C_test.shape[0]),
                                                'pc_model_obs_expc1_expc2': np.round(
                                                    len(np.where(x_model_obs_expc1_expc2 == 1)[0]) / len(
                                                        x_model_obs_expc1_expc2), 1),
                                                'pc_model_obs_expc1': np.round(
                                                    len(np.where(x_model_obs_expc1 == 1)[0]) / len(x_model_obs_expc1),
                                                    1),
                                                'pc_model_obs_expc2': np.round(
                                                    len(np.where(x_model_obs_expc2 == 1)[0]) / len(x_model_obs_expc2),
                                                    1),
                                                'pc_model_obs': np.round(
                                                    len(np.where(x_model_obs == 1)[0]) / len(x_model_obs), 1),
                                                'pc_behavior': np.round(
                                                    len(np.where(x_behavior == 1)[0]) / len(x_behavior), 1),
                                                'x_model': x_model_obs_expc1_expc2,
                                                'x_behavior': x_behavior,
                                                'y_model': y_model_obs_expc1_expc2, 'y_behavior': y_behavior,
                                                'mean_y_behavior': mean_y_behavior[th_count],
                                                'mean_y_model_obs_expc1_expc2': mean_y_model_obs_expc1_expc2[th_count],
                                                'mean_y_model_obs_expc1': mean_y_model_obs_expc1[th_count],
                                                'mean_y_model_obs_expc2': mean_y_model_obs_expc2[th_count],
                                                'mean_y_model_obs': mean_y_model_obs[th_count],
                                                'mean_y_behavior_select_top': mean_y_behavior_select_top[th_count],
                                                'mean_y_model_obs_expc1_expc2_select_top':
                                                    mean_y_model_obs_expc1_expc2_select_top[th_count],
                                                'mean_y_model_obs_expc1_select_top': mean_y_model_obs_expc1_select_top[
                                                    th_count],
                                                'mean_y_model_obs_expc2_select_top': mean_y_model_obs_expc2_select_top[
                                                    th_count],
                                                'mean_y_model_obs_select_top': mean_y_model_obs_select_top[th_count],
                                                'mean_y_model_obs_expc1_expc2_ipw': mean_y_model_obs_expc1_expc2_ipw,
                                                'mean_y_model_obs_expc1_ipw': mean_y_model_obs_expc1_ipw,
                                                'mean_y_model_obs_expc2_ipw': mean_y_model_obs_expc2_ipw,
                                                'mean_y_model_obs_ipw': mean_y_model_obs_ipw,
                                                'mean_y_behavior_ipw': mean_y_behavior_ipw,
                                                'cv': cv},
                                               ignore_index=True)

                print(
                    "th: %.3f, mean_y_model: %.3f, mean_y_behavior: %.3f" % (th, mean_y_model_obs_expc1_expc2[th_count],
                                                                             mean_y_behavior[th_count]))

        plot_rollout_eval(rollout_df, data_name=data_name, maxcv=maxcv, eval_prop=1, eval_sort=True)
        rollout_df.to_pickle(os.path.join('./results', data_name, 'rollout_df.pkl'))


def learn_data_bounds(data_dict, train_all=False, cv=0, train_idx=None, data_name='IST', n_total=500):
    """
    Learn IST bounds for the given dataset
    :param n_total:
    :param train_idx: train index for consistent cross validation
    :param cv: cv id
    :param train_all: if True, train all intermediate models
    :param data_dict: dictionary containing the data
    :return: IST bounds
    """
    data_str = data_name + '_bounds'
    if train_idx is None:
        train_idx = range(data_dict['train_df_obs'].shape[0])

    c1_dims = data_dict['c1_dims']
    c2_dims = data_dict['c2_dims']
    treatment_dim = data_dict['treatment']
    outcome_dim = data_dict['target']

    train_df_obs_this = data_dict['train_df_obs'].iloc[train_idx]
    test_df_obs_this = data_dict['test_df_obs']
    train_df_expc1_this = data_dict['train_df_expc1'].iloc[train_idx]
    train_df_expc2_this = data_dict['train_df_expc2'].iloc[train_idx]

    # one hot encoding of the treatment and outcome
    y_obs = np.asarray(train_df_obs_this[outcome_dim])
    y_expc1 = np.asarray(train_df_expc1_this[outcome_dim])
    y_expc2 = np.asarray(train_df_expc2_this[outcome_dim])

    x_obs = np.asarray(train_df_obs_this[treatment_dim])
    x_card = np.unique(x_obs)
    y_card = np.unique(y_obs)

    # IST bounds py_givenxc1c2
    if train_all:
        print('hre')
        model_pygivenxc1c2 = train_wrapper_bounds(X=np.asarray(train_df_obs_this[c1_dims + c2_dims + treatment_dim]),
                                                  targets=y_obs,
                                                  lower_bounds=None,
                                                  data_name=data_str + '_py_givenxc1c2', loss_type='cross_entropy',
                                                  model='XGBoost', cv=cv)
    else:
        model_pygivenxc1c2 = XGBModel(input_size=len(c2_dims) + len(c1_dims) + len(treatment_dim), target_size=2)
        model_pygivenxc1c2.load(
            os.path.join('./ckpt', data_str + '_py_givenxc1c2' + '_bounds', 'xXGBoostModel_%d.pkl' % cv))

    y_train_obsc1c2_logits = np.zeros((train_df_obs_this.shape[0], len(x_card)))
    y_test_obsc1c2_logits = np.zeros((test_df_obs_this.shape[0], len(x_card)))
    for ii, xx in enumerate(x_card):
        x1_train = np.concatenate(
            (np.asarray(train_df_obs_this[c1_dims + c2_dims]), xx * np.ones((train_df_obs_this.shape[0], 1))),
            axis=1)
        x1_test = np.concatenate(
            (np.asarray(test_df_obs_this[c1_dims + c2_dims]), xx * np.ones((test_df_obs_this.shape[0], 1))),
            axis=1)
        y_train_obsc1c2_logits[:, ii] = model_pygivenxc1c2.predict(x1_train)[:, 1]
        y_test_obsc1c2_logits[:, ii] = model_pygivenxc1c2.predict(x1_test)[:, 1]

    print('done evaluating_' + data_str + '_py_givenxc1c2')

    # IST bounds py_givenxc1
    if train_all:
        model_pygivenxc1 = train_wrapper_bounds(X=np.asarray(train_df_obs_this[c1_dims + treatment_dim]),
                                                targets=y_obs,
                                                lower_bounds=None,
                                                data_name=data_str + '_py_givenxc1', hidden_size=hidden_size_pygivenxc1,
                                                loss_type='cross_entropy', model='XGBoost', cv=cv)
    else:
        model_pygivenxc1 = XGBModel(input_size=len(c1_dims) + len(treatment_dim), target_size=2)
        model_pygivenxc1.load(
            os.path.join('./ckpt', data_str + '_py_givenxc1' + '_bounds', 'xXGBoostModel_%d.pkl' % cv))

    y_train_obsc1_logits = np.zeros((train_df_obs_this.shape[0], len(x_card)))
    y_test_obsc1_logits = np.zeros((test_df_obs_this.shape[0], len(x_card)))
    for ii, xx in enumerate(x_card):
        x1_train = np.concatenate(
            (np.asarray(train_df_obs_this[c1_dims]), xx * np.ones((train_df_obs_this.shape[0], 1))),
            axis=1)
        x1_test = np.concatenate(
            (np.asarray(test_df_obs_this[c1_dims]), xx * np.ones((test_df_obs_this.shape[0], 1))),
            axis=1)
        y_train_obsc1_logits[:, ii] = model_pygivenxc1.predict(x1_train)[:, 1]
        y_test_obsc1_logits[:, ii] = model_pygivenxc1.predict(x1_test)[:, 1]

    print('done evaluating IST bounds py_givenxc1')

    # IST bounds py_givenxc2
    if train_all:
        model_pygivenxc2 = train_wrapper_bounds(X=np.asarray(train_df_obs_this[c2_dims + treatment_dim]),
                                                targets=y_obs,
                                                lower_bounds=None,
                                                data_name=data_str + '_py_givenxc2', hidden_size=hidden_size_pygivenxc2,
                                                loss_type='cross_entropy', model='XGBoost', cv=cv)
    else:
        model_pygivenxc2 = XGBModel(input_size=len(c2_dims) + len(treatment_dim), target_size=2)
        model_pygivenxc2.load(
            os.path.join('./ckpt', data_str + '_py_givenxc2' + '_bounds', 'xXGBoostModel_%d.pkl' % cv))

    y_train_obsc2_logits = np.zeros((train_df_obs_this.shape[0], len(x_card)))
    y_test_obsc2_logits = np.zeros((test_df_obs_this.shape[0], len(x_card)))
    for ii, xx in enumerate(x_card):
        x1_train = np.concatenate(
            (np.asarray(train_df_obs_this[c2_dims]), xx * np.ones((train_df_obs_this.shape[0], 1))),
            axis=1)
        x1_test = np.concatenate(
            (np.asarray(test_df_obs_this[c2_dims]), xx * np.ones((test_df_obs_this.shape[0], 1))),
            axis=1)
        y_train_obsc2_logits[:, ii] = model_pygivenxc2.predict(x1_train)[:, 1]
        y_test_obsc2_logits[:, ii] = model_pygivenxc2.predict(x1_test)[:, 1]

    print('done evaluating ' + data_str + ' py_givenxc2')

    # IST bounds py_doxc1
    if train_all:
        model_py_dox_givenxc1 = train_wrapper_bounds(X=np.asarray(train_df_expc1_this[c1_dims + treatment_dim]),
                                                     targets=y_expc1,
                                                     lower_bounds=None,
                                                     data_name='IST_bounds_py_givendoxc1', loss_type='cross_entropy',
                                                     model='XGBoost', cv=cv)
    else:
        model_py_dox_givenxc1 = XGBModel(input_size=len(c1_dims) + len(treatment_dim), target_size=2)
        model_py_dox_givenxc1.load(
            os.path.join('./ckpt', data_str + '_py_givendoxc1' + '_bounds', 'xXGBoostModel_%d.pkl' % cv))

    y_dox_train_expc1_logits = np.zeros((train_df_obs_this.shape[0], len(x_card)))
    y_dox_test_expc1_logits = np.zeros((test_df_obs_this.shape[0], len(x_card)))
    for ii, xx in enumerate(x_card):
        x1_train = np.concatenate(
            (np.asarray(train_df_obs_this[c1_dims]), xx * np.ones((train_df_obs_this.shape[0], 1))),
            axis=1)
        x1_test = np.concatenate(
            (np.asarray(test_df_obs_this[c1_dims]), xx * np.ones((test_df_obs_this.shape[0], 1))),
            axis=1)
        y_dox_train_expc1_logits[:, ii] = model_py_dox_givenxc1.predict(x1_train)[:, 1]
        y_dox_test_expc1_logits[:, ii] = model_py_dox_givenxc1.predict(x1_test)[:, 1]

    print('done evaluating ' + data_str + ' py_doxc1')

    # IST bounds py_doxc2
    if train_all:
        model_py_dox_givenxc2 = train_wrapper_bounds(X=np.asarray(train_df_expc2_this[c2_dims + treatment_dim]),
                                                     targets=y_expc2,
                                                     lower_bounds=None,
                                                     data_name=data_str + '_py_givendoxc2', loss_type='cross_entropy',
                                                     model='XGBoost', cv=cv)
    else:
        model_py_dox_givenxc2 = XGBModel(input_size=len(c2_dims) + len(treatment_dim), target_size=2)
        model_py_dox_givenxc2.load(
            os.path.join('./ckpt', data_str + '_py_givendoxc2' + '_bounds', 'xXGBoostModel_%d.pkl' % cv))

    y_dox_train_expc2_logits = np.zeros((train_df_obs_this.shape[0], len(x_card)))
    y_dox_test_expc2_logits = np.zeros((test_df_obs_this.shape[0], len(x_card)))
    for ii, xx in enumerate(x_card):
        x1_train = np.concatenate(
            (np.asarray(train_df_obs_this[c2_dims]), xx * np.ones((train_df_obs_this.shape[0], 1))),
            axis=1)
        x1_test = np.concatenate(
            (np.asarray(test_df_obs_this[c2_dims]), xx * np.ones((test_df_obs_this.shape[0], 1))),
            axis=1)
        y_dox_train_expc2_logits[:, ii] = model_py_dox_givenxc2.predict(x1_train)[:, 1]
        y_dox_test_expc2_logits[:, ii] = model_py_dox_givenxc2.predict(x1_test)[:, 1]

    print('done evaluating ' + data_str + ' py_doxc2')

    # IST bounds py_x_joint_obs
    y_card = np.unique(y_obs)
    x_card = np.unique(x_obs)

    py_x_joint_obs = np.zeros((len(y_card), len(x_card)))
    for i, yy in enumerate(y_card):
        for j, xx in enumerate(x_card):
            py_x_joint_obs[i, j] = np.sum((y_obs == yy) & (x_obs == xx))

    py_x_joint_obs = py_x_joint_obs / np.sum(py_x_joint_obs, keepdims=True)

    # IST bounds py_dox_marginal_exp
    py_dox_marginal_exp = np.zeros((len(y_card), len(x_card)))

    x_expc1 = np.asarray(train_df_expc1_this[treatment_dim])
    x_expc2 = np.asarray(train_df_expc2_this[treatment_dim])

    for i, yy in enumerate(y_card):
        for j, xx in enumerate(x_card):
            py_dox_marginal_exp[i, j] = np.sum((y_expc1 == yy) & (x_expc1 == xx))

    py_dox_marginal_exp = py_dox_marginal_exp / np.sum(py_dox_marginal_exp, axis=0, keepdims=True)

    # IST bounds px_givenc1c2
    if train_all:
        # print('no of treatments', np.unique(x_obs))
        model_pxgivenc1c2 = train_wrapper_bounds(X=np.asarray(train_df_obs_this[c1_dims + c2_dims]), targets=x_obs,
                                                 lower_bounds=None,
                                                 data_name=data_str + '_px_givenc1c2', loss_type='cross_entropy',
                                                 n_targets=len(x_card), n_epochs=30,
                                                 hidden_size=hidden_size_px_givenc1c2, model='XGBoost', cv=cv)
    else:
        model_pxgivenc1c2 = XGBModel(input_size=len(c1_dims) + len(c2_dims), target_size=len(x_card))
        model_pxgivenc1c2.load(
            os.path.join('./ckpt', data_str + '_px_givenc1c2' + '_bounds', 'xXGBoostModel_%d.pkl' % cv))

    x_train_obsc1c2_logits = model_pxgivenc1c2.predict(np.asarray(train_df_obs_this[c1_dims + c2_dims]))
    x_test_obsc1c2_logits = model_pxgivenc1c2.predict(np.asarray(test_df_obs_this[c1_dims + c2_dims]))

    print('done evaluating %s bounds px_givenc1c2' % data_str)

    # IST bounds px_givenc1
    if train_all:
        model_pxgivenc1 = train_wrapper_bounds(X=np.asarray(train_df_obs_this[c1_dims]), targets=x_obs,
                                               lower_bounds=None,
                                               data_name=data_str + '_px_givenc1', loss_type='cross_entropy',
                                               n_targets=len(x_card), n_epochs=30, hidden_size=hidden_size_px_givenc1c2,
                                               model='XGBoost', cv=cv)
    else:
        model_pxgivenc1 = XGBModel(input_size=len(c1_dims), target_size=len(x_card))
        model_pxgivenc1.load(os.path.join('./ckpt', data_str + '_px_givenc1' + '_bounds', 'xXGBoostModel_%d.pkl' % cv))

    x_train_obsc1_logits = model_pxgivenc1.predict(np.asarray(train_df_obs_this[c1_dims]))
    x_test_obsc1_logits = model_pxgivenc1.predict(np.asarray(test_df_obs_this[c1_dims]))

    print('done evaluating %s bounds px_givenc1' % data_str)

    # IST bounds px_givenc2
    if train_all:
        model_pxgivenc2 = train_wrapper_bounds(X=np.asarray(train_df_obs_this[c2_dims]), targets=x_obs,
                                               lower_bounds=None,
                                               data_name=data_str + '_px_givenc2', loss_type='cross_entropy',
                                               n_targets=len(x_card), n_epochs=30, hidden_size=hidden_size_px_givenc1c2,
                                               model='XGBoost', cv=cv)
    else:
        model_pxgivenc2 = XGBModel(input_size=len(c2_dims), target_size=len(x_card))
        model_pxgivenc2.load(os.path.join('./ckpt', data_str + '_px_givenc2' + '_bounds', 'xXGBoostModel.pkl' % cv))

    x_train_obsc2_logits = model_pxgivenc2.predict(np.asarray(train_df_obs_this[c2_dims]))
    x_test_obsc2_logits = model_pxgivenc2.predict(np.asarray(test_df_obs_this[c2_dims]))

    print('done evaluating ' + data_str + ' px_givenc2')

    # IST bounds pc1_givenc2 and pc2_givenc1
    if train_all:
        if data_name == "IST":
            model_pc1givenc2, model_pc2_givenc1 = train_wrapper_ist_c1c2(C1_data=np.asarray(train_df_obs_this[c1_dims]),
                                                                         C2_data=np.asarray(train_df_obs_this[c2_dims]),
                                                                         loss_type=['binary_cross_entropy',
                                                                                    'mean_squared_error'],
                                                                         data_name=data_str,
                                                                         hidden_size=hidden_size_pc1c2, n_epochs=45,
                                                                         cv=cv)
        else:
            model_pc1givenc2, model_pc2_givenc1 = train_wrapper_ist_c1c2(C1_data=np.asarray(train_df_obs_this[c1_dims]),
                                                                         C2_data=np.asarray(train_df_obs_this[c2_dims]),
                                                                         loss_type=['mean_squared_error',
                                                                                    'mean_squared_error'],
                                                                         data_name=data_str,
                                                                         hidden_size=hidden_size_pc1c2, n_epochs=45,
                                                                         cv=cv)
    else:
        model_pc1givenc2 = NNmodel(input_size=len(c2_dims), target_size=len(c1_dims), hidden_size=hidden_size_pc1c2)
        model_pc1givenc2.load_state_dict(
            torch.load(os.path.join('./ckpt', data_str + '_c1_givenc2', '_%d.pt' % cv)))
        model_pc2_givenc1 = NNmodel(input_size=len(c1_dims), target_size=len(c2_dims) * 2,
                                    hidden_size=hidden_size_pc1c2)
        model_pc2_givenc1.load_state_dict(
            torch.load(os.path.join('./ckpt', data_str + '_c2_givenc1', '_%d.pt' % cv)))

    model_pc1givenc2.eval()
    model_pc2_givenc1.eval()

    if data_name == "IST":
        pc1_givenc2_train_logits = np.zeros((train_df_obs_this.shape[0], len(c1_dims)))
        pc1_givenc2_test_logits = np.zeros((test_df_obs_this.shape[0], len(c1_dims)))

        for i in range(len(c1_dims)):
            idxs = np.where(np.asarray(train_df_obs_this[c1_dims[i]]) == 1)[0]
            p_train = idxs.shape[0] / train_df_obs_this.shape[0]
            pc1_givenc2_train_logits[idxs, i] = p_train
            pc1_givenc2_train_logits[np.setdiff1d(range(train_df_obs_this.shape[0]), idxs), i] = 1 - p_train

            idxs = np.where(np.asarray(test_df_obs_this[c1_dims[i]]) == 1)[0]
            pc1_givenc2_test_logits[idxs, i] = p_train
            pc1_givenc2_test_logits[np.setdiff1d(range(test_df_obs_this.shape[0]), idxs), i] = 1 - p_train

        c2_mat = np.asarray(train_df_obs_this[c2_dims])
        c1_mat = np.asarray(train_df_obs_this[c1_dims])
        c2_mat_test = np.asarray(test_df_obs_this[c2_dims])
        c1_mat_test = np.asarray(test_df_obs_this[c1_dims])

        # assumes age and sex are independent
        pc1_givenc2_train_logits = np.prod(pc1_givenc2_train_logits, axis=1)
        pc1_givenc2_test_logits = np.prod(pc1_givenc2_test_logits, axis=1)
    else:
        c2_mat = np.asarray(train_df_obs_this[c2_dims])
        c1_mat = np.asarray(train_df_obs_this[c1_dims])
        c2_mat_test = np.asarray(test_df_obs_this[c2_dims])
        c1_mat_test = np.asarray(test_df_obs_this[c1_dims])

        pc1_givenc2_train_logits = model_pc1givenc2(
            torch.Tensor(np.asarray(train_df_obs_this[c2_dims])).float()).detach().numpy()
        pc1_givenc2_test_logits = model_pc1givenc2(
            torch.Tensor(np.asarray(test_df_obs_this[c2_dims])).float()).detach().numpy()

        scale = np.sqrt(np.var(c1_mat[:, 0] - pc1_givenc2_train_logits[:, 0]))

        pc1_givenc2_train_logits = np.asarray(
            [norm.pdf(x=c1_mat[i, 0], loc=pc1_givenc2_train_logits[i, 0], scale=scale) for i in
             range(len(c1_mat))])
        pc1_givenc2_test_logits = np.asarray(
            [norm.pdf(x=c1_mat_test[i, 0], loc=pc1_givenc2_test_logits[i, 0], scale=scale) for i in
             range(len(c1_mat_test))])

    pc2_givenc1_train_logits = model_pc2_givenc1(
        torch.Tensor(np.asarray(train_df_obs_this[c1_dims])).float()).detach().numpy()
    pc2_givenc1_test_logits = model_pc2_givenc1(
        torch.Tensor(np.asarray(test_df_obs_this[c1_dims])).float()).detach().numpy()

    scale = np.sqrt(np.var(c2_mat[:, 0] - pc2_givenc1_train_logits[:, 0]))

    pc2_givenc1_train_logits = np.asarray(
        [norm.pdf(x=c2_mat[i, 0], loc=pc2_givenc1_train_logits[i, 0], scale=scale) for i in
         range(len(c2_mat))])
    pc2_givenc1_test_logits = np.asarray(
        [norm.pdf(x=c2_mat_test[i, 0], loc=pc2_givenc1_test_logits[i, 0], scale=scale) for i in
         range(len(c2_mat_test))])

    mean_c2 = np.mean(np.asarray(train_df_obs_this[c2_dims]), axis=0)
    std_c2 = np.std(np.asarray(train_df_obs_this[c2_dims]), axis=0)
    p_c2_marginal_train = norm.pdf(x=np.asarray(train_df_obs_this[c2_dims]).reshape(-1), loc=mean_c2, scale=std_c2)
    p_c2_marginal_test = norm.pdf(x=np.asarray(test_df_obs_this[c2_dims]).reshape(-1), loc=mean_c2, scale=std_c2)

    print('done evaluating %s bounds pc1givenc2 and pc2givenc1' % data_str)

    manski_lb_train = np.zeros((train_df_obs_this.shape[0], len(x_card)))
    manski_lb_test = np.zeros((test_df_obs_this.shape[0], len(x_card)))

    int_expc1_lb_train = np.zeros((train_df_obs_this.shape[0], len(x_card)))
    int_expc1_lb_test = np.zeros((test_df_obs_this.shape[0], len(x_card)))

    int_expc2_lb_train = np.zeros((train_df_obs_this.shape[0], len(x_card)))
    int_expc2_lb_test = np.zeros((test_df_obs_this.shape[0], len(x_card)))

    int_expc1_expc2_lb_train = np.zeros((train_df_obs_this.shape[0], len(x_card)))
    int_expc1_expc2_lb_test = np.zeros((test_df_obs_this.shape[0], len(x_card)))

    for ii, xx in enumerate(x_card):
        manski_lb_train[:, ii], _ = manski_bounds(y_train_obsc1c2_logits[:, ii], x_train_obsc2_logits[:, ii])
        manski_lb_test[:, ii], _ = manski_bounds(y_test_obsc1c2_logits[:, ii], x_test_obsc2_logits[:, ii])

        int_expc1_lb_train[:, ii], _ = int_bound_expc1(p_y_given_x_c1_c2=y_train_obsc1c2_logits[:, ii],
                                                       p_x_given_c1_c2=x_train_obsc1c2_logits[:, ii],
                                                       p_ydox_given_x_c1=y_dox_train_expc1_logits[:, ii],
                                                       p_y_given_x_c1=y_train_obsc1_logits[:, ii],
                                                       p_x_given_c1=x_train_obsc1_logits[:, ii],
                                                       p_c2_given_c1=pc2_givenc1_train_logits)

        int_expc1_lb_test[:, ii], _ = int_bound_expc1(p_y_given_x_c1_c2=y_test_obsc1c2_logits[:, ii],
                                                      p_x_given_c1_c2=x_test_obsc1c2_logits[:, ii],
                                                      p_ydox_given_x_c1=y_dox_test_expc1_logits[:, ii],
                                                      p_y_given_x_c1=y_test_obsc1_logits[:, ii],
                                                      p_x_given_c1=x_test_obsc1_logits[:, ii],
                                                      p_c2_given_c1=pc2_givenc1_test_logits)

        int_expc2_lb_train[:, ii], _ = int_bound_expc1(p_y_given_x_c1_c2=y_train_obsc1c2_logits[:, ii],
                                                       p_x_given_c1_c2=x_train_obsc1c2_logits[:, ii],
                                                       p_ydox_given_x_c1=y_dox_train_expc2_logits[:, ii],
                                                       p_y_given_x_c1=y_train_obsc2_logits[:, ii],
                                                       p_x_given_c1=x_train_obsc2_logits[:, ii],
                                                       p_c2_given_c1=pc1_givenc2_train_logits)

        int_expc2_lb_test[:, ii], _ = int_bound_expc1(p_y_given_x_c1_c2=y_test_obsc1c2_logits[:, ii],
                                                      p_x_given_c1_c2=x_test_obsc1c2_logits[:, ii],
                                                      p_ydox_given_x_c1=y_dox_test_expc2_logits[:, ii],
                                                      p_y_given_x_c1=y_test_obsc2_logits[:, ii],
                                                      p_x_given_c1=x_test_obsc2_logits[:, ii],
                                                      p_c2_given_c1=pc1_givenc2_test_logits)

        int_expc1_expc2_lb_train[:, ii], _ = int_bounds_expc1_expc2(
            p_y_x=py_x_joint_obs[1, ii] * np.ones(train_df_obs_this.shape[0]),
            p_y_dox=py_dox_marginal_exp[1, ii] * np.ones(train_df_obs_this.shape[0]),
            p_y_dox_given_c1=y_dox_train_expc1_logits[:, ii],
            p_y_dox_given_c2=y_dox_train_expc2_logits[:, ii],
            p_c2_given_c1=pc2_givenc1_train_logits,
            p_c1_given_c2=pc1_givenc2_train_logits,
            p_c1_c2=np.multiply(
                pc1_givenc2_train_logits,
                p_c2_marginal_train),
            p_y_given_x_c1_c2=y_train_obsc1c2_logits[:, ii],
            p_x_given_c1_c2=x_train_obsc1c2_logits[:, ii],
            p_y_given_x_c1=y_train_obsc1_logits[:, ii],
            p_y_given_x_c2=y_train_obsc2_logits[:, ii],
            p_x_given_c1=x_train_obsc1_logits[
                         :, ii],
            p_x_given_c2=x_train_obsc2_logits[
                         :, ii])

        int_expc1_expc2_lb_test[:, ii], _ = int_bounds_expc1_expc2(
            p_y_x=py_x_joint_obs[1, ii] * np.ones(test_df_obs_this.shape[0]),
            p_y_dox=py_dox_marginal_exp[1, ii] * np.ones(test_df_obs_this.shape[0]),
            p_y_dox_given_c1=y_dox_test_expc1_logits[:, ii],
            p_y_dox_given_c2=y_dox_test_expc2_logits[:, ii],
            p_c2_given_c1=pc2_givenc1_test_logits,
            p_c1_given_c2=pc1_givenc2_test_logits,
            p_c1_c2=np.multiply(
                pc1_givenc2_test_logits,
                p_c2_marginal_test),
            p_y_given_x_c1_c2=y_test_obsc1c2_logits[:, ii],
            p_x_given_c1_c2=x_test_obsc1c2_logits[:, ii],
            p_y_given_x_c1=y_test_obsc1_logits[:, ii],
            p_y_given_x_c2=y_test_obsc2_logits[:, ii],
            p_x_given_c1=x_test_obsc1_logits[
                         :, ii],
            p_x_given_c2=x_test_obsc2_logits[
                         :, ii])

    manski_lb_train = np.clip(manski_lb_train, a_min=0, a_max=1)
    manski_lb_test = np.clip(manski_lb_test, a_min=0, a_max=1)

    int_expc1_lb_train = np.clip(int_expc1_lb_train, a_min=0, a_max=1)
    int_expc1_lb_test = np.clip(int_expc1_lb_test, a_min=0, a_max=1)

    int_expc2_lb_train = np.clip(int_expc2_lb_train, a_min=0, a_max=1)
    int_expc2_lb_test = np.clip(int_expc2_lb_test, a_min=0, a_max=1)

    int_expc1_expc2_lb_train = np.clip(int_expc1_expc2_lb_train, a_min=0, a_max=1)
    int_expc1_expc2_lb_test = np.clip(int_expc1_expc2_lb_test, a_min=0, a_max=1)

    weight_matrix_train = np.zeros((train_df_obs_this.shape[0], 4, len(x_card)))
    weight_matrix_train[:, 0, :] = manski_lb_train
    weight_matrix_train[:, 1, :] = int_expc1_lb_train
    weight_matrix_train[:, 2, :] = int_expc2_lb_train
    weight_matrix_train[:, 3, :] = int_expc1_expc2_lb_train

    weight_matrix_test = np.zeros((test_df_obs_this.shape[0], 4, len(x_card)))
    weight_matrix_test[:, 0, :] = manski_lb_test
    weight_matrix_test[:, 1, :] = int_expc1_lb_test
    weight_matrix_test[:, 2, :] = int_expc2_lb_test
    weight_matrix_test[:, 3, :] = int_expc1_expc2_lb_test

    data_dict['weight_matrix_train_%d' % cv] = weight_matrix_train
    data_dict['weight_matrix_test_%d' % cv] = weight_matrix_test

    with open('./data/' + data_name + '_data_dict_n_%d_d_%d.pkl' % (n_total, len(c1_dims) + len(c2_dims)), 'wb') as f:
        pkl.dump(data_dict, f)

    return weight_matrix_train, weight_matrix_test
