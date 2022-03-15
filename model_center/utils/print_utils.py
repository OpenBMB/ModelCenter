import torch
import bmtrain as bmt

def print_inspect(model : torch.nn.Module, param_name : str, prefix : str = ''):
    """Inspect the model and print the summary of the parameters on rank 0.

    Args:
        model (torch.nn.Module): The model to be inspected.
        param_name (str): The name of the parameter to be inspected. The wildcard '*' can be used to match multiple parameters.
        prefix (str): The prefix of the parameter name.
    
    Example:
        >>> from model_center.utils import print_inspect
        >>> print_inspect(model, "*.linear*")
        name   shape     max     min     std     mean    grad_std  grad_mean
        ...

    """
    bmt.print_rank(
        bmt.inspect.format_summary(
            bmt.inspect.inspect_model(model, param_name, prefix)
        )
    )