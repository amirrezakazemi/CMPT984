import torch
import pandas as pd


def generate_data(input_dir, output_dir):
    """

    :param input_dir: directory of the generative model
    :param output_dir: directory of generated synthetic data
    :return:
    """
    model = torch.load(input_dir)
    synthetic_data = pd.DataFrame(model.generate((1000, 20)).detach().numpy())
    label_data("models/TrainedFConBortezomib.pt", synthetic_data, output_dir)
    #synthetic_data.to_csv(output_dir, sep="\t")


def label_data(input_dir, data, output_dir):
    """

    :param input_dir: directory of the predictive model
    :param data: synthetic data
    :param output_dir: directory of the saved synthetic data
    :return:
    """
    model = torch.load(input_dir)
    labels = pd.DataFrame(model(torch.tensor(data.values)).detach().numpy())
    synthetic_data = pd.concat([data, labels], axis=1)
    synthetic_data.to_csv(output_dir, sep="\t")


generate_data("models/TrainedVAEonBortezomib.pt", "synthetic_data/VAEGeneratedData.tsv")


