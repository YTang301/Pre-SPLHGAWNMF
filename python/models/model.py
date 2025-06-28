from .SPLHGAWNMF import vanillaSPLHyperModel
import numpy as np

def model(interaction, sd, sm, modelName='SPLHGAWNMF'):
    if modelName == 'SPLHGAWNMF':
        return np.mat(vanillaSPLHyperModel(interaction, sd, sm))
    else:
        print("Model not found!")
