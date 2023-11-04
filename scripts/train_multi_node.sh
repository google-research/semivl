#!/bin/bash
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


now=$(date +"%Y%m%d_%H%M%S")
port=$(($RANDOM + 10000))

method=$1
config=$2
master_addr=$3
master_port=$4

echo 'Start training' $method 'with' $config 'on' $master_addr $port
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$JOB_COMPLETION_INDEX \
    --nproc_per_node=$NGPUS \
    --master_addr=$master_addr \
    --master_port=$master_port \
    $method.py --config=$config --port $master_port 2>&1
