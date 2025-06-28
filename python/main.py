import pandas as pd
import os
import sys
import argparse
from time import strftime, time
# from utils import Logger
from loadData import loadData
from featureEngineer import integrateSimilarity
from models.model import model
from copy import deepcopy


path = os.path.abspath(".")
sys.path.append(path)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-ds", default="HDVD", help="Dataset to use")
parser.add_argument(
    "--featureEngineer", "-fe", default=None, help="Feature engineer to use"
)
parser.add_argument("--modelName", "-m", default="SPLHGAWNMF", help="Model name to use")
parser.add_argument(
    "--message", "-msg", default="No message", help="Message to save in the log file"
)
args = parser.parse_args()
dataset = args.dataset
featureEngineer = args.featureEngineer
modelName = args.modelName
message = args.message

print("Message: ", message, "\n")
startTime = strftime("%Y-%m-%d %H:%M:%S")

print("-" * 60)
print("Dataset:           " + dataset)
print("Feature Engineer:  " + str(featureEngineer))
print("Model:             " + modelName)

print("-" * 60)
print("Working directory: " + os.getcwd())
print("Start time:        " + startTime, "\n")

# Load Data
VDA, interaction, dName, mName, FS, FSP, SS, SSP = loadData(dataset)
interaction0 = deepcopy(interaction)
interaction, sm, sd = integrateSimilarity(
    FS, FSP, SS, SSP, interaction, featureEngineer
)
F = model(interaction, sd, sm, modelName)
score = F[0, :].tolist()[0]
virus = 'SARS-CoV-2'  # Assuming the first row corresponds to SARS-CoV-2
df = pd.DataFrame({'SARS-CoV-2': mName, 'score': score})
df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
df.to_csv('./Result/CaseStudy_' + dataset + '_' + modelName + '_' + virus + '.csv')
print("Case Study for ", virus, " is done!")
print("Result saved to ", './Result/CaseStudy_' + dataset + '_' + modelName + '_' + virus + '.csv')
