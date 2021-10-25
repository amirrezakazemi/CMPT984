import torch
from training_vae import runVAE
from training import run
from models.generating_models import *
from models.predicting_models import *
import sys


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    model_type, nepoch, drug = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    model = None
    if model_type == "VAE":
        model = VAE(11609, [5000, 1000, 500, 100], 20, [100, 500, 1000, 5000])
        trained_model = runVAE(model, nepoch)
        torch.save(trained_model, "models/Trained" + model_type + "on" + drug + ".pt")

    elif model_type == "FC":
        model = FC(11609, [5000, 1000, 50, 5], 1)
        trained_model = run(model, nepoch)
        torch.save(trained_model, "models/Trained" + model_type + "on" + drug + ".pt")


