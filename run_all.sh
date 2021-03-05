#!/bin/bash

python main.py models/agent data agent train  --embed_dim=128 --override_modeldir
python main.py models/agent data agent test --embed_dim=128

python main.py models/patient data patient train  --embed_dim=128 --override_modeldir
python main.py models/patient data patient test --embed_dim=128

python main.py models/experiencer data experiencer train  --embed_dim=128 --override_modeldir
python main.py models/experiencer data experiencer test --embed_dim=128

python main.py models/theme data theme train  --embed_dim=128 --override_modeldir
python main.py models/theme data theme test --embed_dim=128

python main.py models/recipient data recipient train  --embed_dim=128 --override_modeldir
python main.py models/recipient data recipient test --embed_dim=128

# Same experiments with upsampled train data for unbalanced datasets

python main.py models/agent-upsampled data agent train  --embed_dim=128 --override_modeldir --upsampled_data
python main.py models/agent-upsampled data agent test --embed_dim=128

python main.py models/patient-upsampled data patient train  --embed_dim=128 --override_modeldir --upsampled_data
python main.py models/patient-upsampled data patient test --embed_dim=128

python main.py models/experiencer-upsampled data experiencer train  --embed_dim=128 --override_modeldir --upsampled_data
python main.py models/experiencer-upsampled data experiencer test --embed_dim=128

python main.py models/theme-upsampled data theme train  --embed_dim=128 --override_modeldir --upsampled_data
python main.py models/theme-upsampled data theme test --embed_dim=128

python main.py models/recipient-upsampled data recipient train  --embed_dim=128 --override_modeldir --upsampled_data
python main.py models/recipient-upsampled data recipient test --embed_dim=128 
