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


import os
import random

if __name__ == '__main__':
    root = os.path.expanduser('~/data/ADEChallengeData2016/')

    files = sorted(os.listdir(root + 'images/training'))
    random.Random(0).shuffle(files)
    n_files = len(files)

    def save_split(file_name, selected, mode='training'):
        out_str = ""
        for s in selected:
            s = s.rsplit('.', 1)[0]
            out_str += f'images/{mode}/{s}.jpg annotations/{mode}/{s}.png\n'
        out_str = out_str[:-1]  # remove last \n
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as f:
            f.write(out_str)

    for split in [128, 64, 32, 16, 8]:
        n_split = round(n_files / split)
        print(f'Split 1_{split}')
        print(f'Select {n_split} of {n_files}')
        labeled = files[:n_split]
        unlabeled = files[n_split:]
        assert len(labeled) + len(unlabeled) == n_files
        save_split(f'splits/ade/1_{split}/labeled.txt', labeled)
        save_split(f'splits/ade/1_{split}/unlabeled.txt', unlabeled)

    files = sorted(os.listdir(root + 'images/validation'))
    save_split(f'splits/ade/val.txt', files, mode='validation')