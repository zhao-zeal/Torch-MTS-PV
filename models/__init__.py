
from .DLinear import DLinear



def model_select(name):
    name = name.upper()

    if name == "DLINEAR":
        return DLinear
    elif name == "PATCHTST":
        return PatchTST

    else:
        raise NotImplementedError
