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
n_gpus=$3

echo 'Start training on port' $port
python -m torch.distributed.launch \
    --nproc_per_node=$n_gpus \
    --master_addr=localhost \
    --master_port=$port \
    $method.py \
    --config=$config --port $port 2>&1
