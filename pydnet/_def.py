import torch
import base64

from ._util import py_d_model
from settings import py_d_net


class PydNet:
    def __init__(self):
        self._model = torch.hub.load(str(py_d_net), base64.b64decode(py_d_model["model"]).decode(), source="local")
        self._transformer = torch.hub.load(str(py_d_net), "transforms", source="local")

    @property
    def transformer(self):
        return self._transformer

    @property
    def model(self):
        return self._model

