import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys


class GenDataset(Dataset):

    def __init__(self, data_dir):
        """
        :param data_dir: directory of data
        """
        self.data = pd.read_csv(data_dir, sep="\t", index_col=0)
        self.X = self.data.to_numpy()
        pass

    def __len__(self):
        return self.X.shape[0]
        pass

    def __getitem__(self, idx):
        return self.X[idx, :]
        pass


class PredDataset(Dataset):
    def __init__(self, data_dir):
        """
        :param data_dir:
        """
        self.data = pd.read_csv(data_dir, sep="\t", index_col=0)
        self.X = self.data.drop(['logIC50', 'response'], axis=1).to_numpy()
        self.Y = self.data['logIC50'].to_numpy()

    def __len__(self):
        return self.X.shape[0]
        pass

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx]
        pass


def create_data(drug, flag1=0, flag2=0):
    """
    :param drug: sys.argv[3] drug_type
    :param flag1: sys.argv[4] if flag1==0: only using source data; else: use source and target data
    :param flag2: sys.argv[5] if flag2==1: create data for predictive model; else: make data for generative model
    :return: dataloader_train, dataloader_test
    """
    data = None
    if flag2:
        data = pd.read_csv("data/split/" + drug + "/Source_exprs_resp_z." + drug + ".tsv", sep="\t",
                           index_col=0)
    else:
        source_X = pd.read_csv("data/split/" + drug + "/Source_exprs_resp_z." + drug + ".tsv", sep="\t",
                               index_col=0).drop(['logIC50', 'response'], axis=1)
        if flag1:
            target_X = pd.read_csv("data/split/" + drug + "/Target_combined_expr_resp_z." + drug + ".tsv", sep="\t",
                                   index_col=0).drop(['response'], axis=1)
            data = pd.concat([source_X, target_X])
        else:
            data = source_X

    train, test = train_test_split(data, test_size=0.3)
    train.to_csv("data/split/" + drug + "/Source_exprs_resp_z." + drug + "_train.tsv", sep="\t")
    test.to_csv("data/split/" + drug + "/Source_exprs_resp_z." + drug + "_test.tsv", sep="\t")

    if flag2:
        dataset_train = PredDataset("data/split/" + drug + "/Source_exprs_resp_z." + drug + "_train.tsv")
        dataset_test = PredDataset("data/split/" + drug + "/Source_exprs_resp_z." + drug + "_test.tsv")
    else:
        dataset_train = GenDataset("data/split/" + drug + "/Source_exprs_resp_z." + drug + "_train.tsv")
        dataset_test = GenDataset("data/split/" + drug + "/Source_exprs_resp_z." + drug + "_test.tsv")

    dataloader_train = DataLoader(dataset_train, batch_size=64)
    dataloader_test = DataLoader(dataset_test, batch_size=64)

    return dataloader_train, dataloader_test


dataloader_train, dataloader_test = None, None
dataloader_train, dataloader_test = create_data(sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))