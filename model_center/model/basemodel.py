# coding=utf-8

import os
from typing import Union
import torch
import bmtrain as bmt
from .config.config import Config
from ..utils import check_web_and_convert_path

class BaseModel(torch.nn.Module):

    _CONFIG_TYPE = Config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike]):
        config = cls._CONFIG_TYPE.from_pretrained(pretrained_model_name_or_path)
        path = check_web_and_convert_path(pretrained_model_name_or_path, 'model')
        model = cls(config)
        bmt.load(model, os.path.join(path, 'pytorch_model.pt'), strict=False)
        return model

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        config = cls._CONFIG_TYPE.from_json_file(json_file)
        model = cls(config)
        bmt.init_parameters(model)
        return model
