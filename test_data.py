from lib.config import config
from lib.datasets import get_dataset
from IPython import embed

config.merge_from_file('experiments/wflw/face_alignment_wflw_hrnet_w18.yaml')
dataset_type = get_dataset(config)

dataset = dataset_type(config, is_train=True)

embed()