# coding=utf-8

import os
from typing import Union
import torch
import bmtrain as bmt
from model.config.config import Config


class BaseModel(torch.nn.Module):

    _CONFIG_TYPE = Config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike]):
        print(cls._CONFIG_TYPE)
        config = cls._CONFIG_TYPE.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        bmt.load(model, os.path.join(pretrained_model_name_or_path, 'pytorch_model.pt'))
        return model

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        config = cls._CONFIG_TYPE.from_json_file(json_file)
        model = cls(config)
        bmt.init_parameters(model)
        return model
