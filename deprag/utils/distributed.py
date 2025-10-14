import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Check if the script is running in a distributed environment."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Get the total number of processes."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """Check if the current process is the main one (rank 0)."""
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> None:
    """Initialize the distributed process group."""
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group(backend=backend)

    # Set the device for the current process
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def cleanup_distributed() -> None:
    """Clean up the distributed process group."""
    if is_distributed():
        dist.destroy_process_group()
