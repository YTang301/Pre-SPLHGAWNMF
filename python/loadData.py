import numpy as np
import pandas as pd
import os


#####################
# Load HDVD dataset #
#####################
def Input_HDVD():
    path = '../dataset/HDVD/'
    interaction = np.loadtxt(path + 'virusdrug.csv', delimiter=',').astype(int)
    vName = pd.read_csv(path + 'viruses.csv', header=None).squeeze()
    dName = pd.read_csv(path + 'drugs.csv', header=None).squeeze()
    VDA = np.column_stack(np.where(interaction == 1)) + 1
    VDA[:, 0], VDA[:, 1] = VDA[:, 1], VDA[:, 0].copy()
    
    SS = np.loadtxt(path + 'virussim.csv', delimiter=',')
    SSP = np.ones_like(SS)
    FS = np.loadtxt(path + 'drugsim.csv', delimiter=',')
    FSP = np.ones_like(FS)

    return VDA, interaction, vName, dName, FS, FSP, SS, SSP



#############
# Load VDA #
#############
def Input_VDA():
    path = '../dataset/VDA/'
    interaction = np.loadtxt(path + 'virusdrug.csv', delimiter=',').astype(int)
    # vName = np.loadtxt(path + 'viruses.csv')
    # dName = np.loadtxt(path + 'drugs.csv')
    vName = pd.read_csv(path + 'viruses.csv', header=None).squeeze()
    dName = pd.read_csv(path + 'drugs.csv', header=None).squeeze()
    VDA = np.column_stack(np.where(interaction == 1)) + 1
    VDA[:, 0], VDA[:, 1] = VDA[:, 1], VDA[:, 0].copy()
    
    SS = np.loadtxt(path + 'virussim.csv', delimiter=',')
    SSP = np.ones_like(SS)
    FS = np.loadtxt(path + 'drugsim.csv', delimiter=',')
    FSP = np.ones_like(FS)

    return VDA, interaction, vName, dName, FS, FSP, SS, SSP

######################
# Load data function #
######################
def loadData(dataType):
    """Load data from different data types."""
    VDA, interaction, vName, dName, FS, FSP, SS, SSP = None, None, None, None, None, None, None, None
    print('-' * 60)
    print(f"Loading data {dataType}...")
    if dataType == 'VDA':
        VDA, interaction, vName, dName, FS, FSP, SS, SSP = Input_VDA()
    elif dataType == 'HDVD':
        VDA, interaction, vName, dName, FS, FSP, SS, SSP = Input_HDVD()
    else:
        print("Data type not found!")
        return None
    
    print("Data Shape: ", '\n',
          "VDA (Known VDAs)                               = ", VDA.shape, '\n',
          "interation (Virus-drug adjacent matrix)        = ", interaction.shape, '\n',
          "vName                                          = ", vName.shape, '\n',
          "dName                                          = ", dName.shape, '\n',
          "FS (Drug chemical structure similarity)        = ", FS.shape, '\n',
          "FSP (weight for FS)                            = ", FSP.shape, '\n',
          "SS (Virus genomic sequence similarity)         = ", SS.shape, '\n',
          "SSP (weight for SS)                            = ", SSP.shape, '\n')

    print(f'   Dataset {dataType} loading is done!\n')
    return VDA, interaction, list(vName), list(dName), FS, FSP, SS, SSP
