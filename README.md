# DMWF

## DMWF: Length-Coupled Representation for Multi-Tab Website Fingerprinting via DNS-over-QUIC

This package contains the experiment code only. 
Public hyperparameter and model-construction defaults are set to `0` as placeholders.

## File descriptions

- `dmwf/lctr.py`: constructs the six-channel length-coupled trace
  representation.
- `dmwf/model.py`: defines the DMWF neural-network structure.
- `dmwf/data.py`: validates and loads NPZ experiment datasets.
- `dmwf/preprocess.py`: converts packet-metadata CSV files into NPZ splits.
- `dmwf/train.py`: implements model training and validation-threshold
  selection.
- `dmwf/evaluate.py`: evaluates saved checkpoints on test data.
- `dmwf/metrics.py`: calculates classification, ranking, and cardinality
  metrics.
- `dmwf/__init__.py`: exposes the package's public classes.

The specific hyperparameters have been indicated in the paper.

## Dataset Availability
The private dataset supporting this study cannot be publicly released due to privacy and copyright restrictions. 
Researchers who require the dataset for academic replication or further research purposes may contact the corresponding author via email to submit an application. 
All legitimate academic requests will be reviewed and responded to promptly.


## E-mail
If you have any question, please feel free to contact us by e-mail (jiangtaozhai@nuist.edu.cn).
