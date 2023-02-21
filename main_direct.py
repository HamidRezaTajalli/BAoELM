import training
from pathlib import Path


training.trainer(exp_num=1, saving_path=Path(), elm_type='mlelm', dataset='mnist', hdlyr_size=[200, 200, 700])
