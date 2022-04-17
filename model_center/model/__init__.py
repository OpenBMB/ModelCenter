# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Model Config
from .config import *

# Model Architecture
from .cpm1 import CPM1, CPM1ForLM
from .cpm2 import CPM2, CPM2ForLM
from .t5 import T5, T5ForLM
from .gpt2 import GPT2, GPT2ForLM
from .gptj import GPTj, GPTjForLM
from .bert import Bert, BertForLM
