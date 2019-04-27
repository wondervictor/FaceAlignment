from lib.config import config, update_config
from lib.datasets import get_dataset

from IPython import embed

config.merge_from_file('experiments/300w/face_alignment_300w_hrnet_w18.yaml')

dataset_type = get_dataset(config)


dataset = dataset_type(config, is_train=True)

embed()