# Pre-SPLHGNMF

Pre-SPLHGNMF is a predictive model designed for virus-drug association prediction. It integrates **Preprocessing**, **Self-Paced Learning (SPL)**, **Hypergraph Regularization**, and **Nonnegative Matrix Factorization (NMF)** to handle noisy biological data and improve predictive performance, especially in cold-start scenarios.

## Project Structure

```markdown
.
├── README.md                 # Project description and usage instructions
├── dataset                   # Dataset directory
│   ├── HDVD                  # HDVD dataset
│   │   ├── drugs.csv         
│   │   ├── drugsim.csv       
│   │   ├── virusdrug.csv     
│   │   ├── viruses.csv       
│   │   └── virussim.csv      
│   └── VDA                   # VDA dataset
│       ├── drugs.csv
│       ├── drugsim.csv
│       ├── virusdrug.csv
│       ├── viruses.csv
│       └── virussim.csv
└── python                    # Source code directory
    ├── Result/               # Folder to store prediction results
    ├── featureEngineer.py    # Feature construction and preprocessing
    ├── loadData.py           # Data loading utilities
    ├── main.py               # Entry point for running experiments
    └── models                # Model implementation
        ├── ConstructHW.py    # Hypergraph weight construction
        ├── SPLHGAWNMF.py     # Main model: Self-paced learning + Hypergraph + NMF
        ├── hypergraph_utils.py # Utility functions for hypergraph operations
        └── model.py          # Core model logic and optimization
```

## Run the Model

Change to the `python` directory:

```bash
cd python
```

Run the main script with arguments:

```bash
python main.py --dataset VDA --featureEngineer LNS --modelName SPLHGAWNMF --message "VDA experiment"
```

Typical options:

- `--dataset`: Dataset to use (`VDA` or `HDVD`)
- `--featureEngineer`: Feature engineering method (e.g., `LNS`)
- `--modelName`: Model name.
