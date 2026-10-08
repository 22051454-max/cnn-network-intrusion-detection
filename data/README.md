# Data

- `sample_flows.csv`: synthetic demo traffic (720 benign flows + 20 of each attack class) in CICIDS2017 column format.
- Real dataset: download **CICIDS2017 MachineLearningCSV.zip** from the Canadian Institute for Cybersecurity
  (https://www.unb.ca/cic/datasets/ids-2017.html), unzip into `data/MachineLearningCVE/`, then run
  `python -m ids.train --data data/MachineLearningCVE --rows-per-class 200000`.
