import torch

from .configuration_manager import ConfigurationManager
from .logging import make_log_dict

__all__ = ["ConfigurationManager", "get_device_info", "make_log_dict"]


def get_device_info():
    """
    Get device information

    Parameters
    ----------
    None

    Returns
    -------
    info : dict
        Device information
    """
    info = {}
    info["torch"] = torch.__version__
    info["device.cuda"] = torch.cuda.is_available()
    info["device.cuda.count"] = torch.cuda.device_count()
    for i in range(torch.cuda.device_count()):
        info[f"device.cuda.name.{i}"] = torch.cuda.get_device_name(i)
        info[f"device.cuda.capability.{i}"] = torch.cuda.get_device_capability(
            i
        )
        info[f"device.cuda.memory.{i}"] = torch.cuda.get_device_properties(
            i
        ).total_memory
        info[f"device.cuda.memory.free.{i}"] = torch.cuda.memory_allocated(i)
        info[f"device.cuda.memory.max.{i}"] = torch.cuda.max_memory_allocated(
            i
        )

    return info
