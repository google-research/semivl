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
import tarfile

def is_source_file(x):
    if x.isdir() or x.name.endswith(('.py', '.sh', '.yml', '.yaml', '.json',  '.ipynb', '.json', '.md', '.txt', 'Dockerfile')) \
            and '.mim' not in x.name and 'training-logs/' not in x.name and 'splits/' not in x.name and 'jobs/' not in x.name and '__pycache__' not in x.name:
        return x
    else:
        return None


def gen_code_archive(out_dir, file='code.tar.gz'):
    archive = os.path.join(out_dir, file)
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    with tarfile.open(archive, mode='w:gz') as tar:
        tar.add('.', filter=is_source_file)
    return archive
