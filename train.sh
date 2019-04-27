#!/bin/bash

set -x
DATA_DIR=$PWD/demo
LOG_DIR=$PWD/demo
MODEL_DIR=$PWD/cls
CFG=NONE
DATAFORM='zip'

pip install yacs --user
pip install hdf5storage --user 

# parsing command line arguments:
while [[ $# > 0 ]]
do
key="$1"

case $key in
    -h|--help)
    echo "Usage: toolkit-execute [run_options]"
    echo "Options:"
    echo "  -c|--cfg <config> - which configuration file to use (default NONE)"
    exit 1
    ;;
    -c|--cfg)
    CFG="$2"
    shift # pass argument
    ;;
    *)
    EXTRA_ARGS="$EXTRA_ARGS $1"
    ;;
esac
shift # past argument or value
done

echo "==> train"
echo $PWD
python  tools/train.py --cfg ${CFG}
# python tools/train.py ${CFG} --gpus 1 
#python -m torch.distributed.launch --nproc_per_node=1 tools/train.py ${CFG} --launcher pytorch \
#                           --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} --data-format ${DATAFORM}
