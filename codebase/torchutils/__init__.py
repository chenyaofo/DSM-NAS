import os
import sys
import argparse

from torch.utils.tensorboard import SummaryWriter

from .distributed import init, is_master
from .log import get_logger, LogExceptionHook, create_code_snapshot, get_diagnostic_info
from .common import DummyClass


__version__ = "v0.2.1-alpha0"

__all__ = [
    "logger",
    "summary_writer",
    "output_directory"
]


def get_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_directory", type=str, default=None)
    args, _ = parser.parse_known_args(argv)
    return args


# automatically detect environment variables to initilize distributed mode
init()
# parse command line args
args = get_args(sys.argv)
output_directory = args.output_directory

if is_master():
    if output_directory is not None:
        os.makedirs(args.output_directory, exist_ok=False)
        create_code_snapshot("code", [".py"], ".", args.output_directory)
        with open(os.path.join(args.output_directory, ".diagnostic_info"), "w") as f:
            f.write(get_diagnostic_info())
    logger = get_logger("project", args.output_directory, "log.txt")
    sys.excepthook = LogExceptionHook(logger)
    if output_directory is None:
        summary_writer = DummyClass()
    else:
        summary_writer = SummaryWriter(args.output_directory)
else:
    logger = DummyClass()
    summary_writer = DummyClass()
