r"""
Modified from https://raw.githubusercontent.com/pytorch/pytorch/v1.7.0/torch/distributed/launch.py

This script aims to quickly start Single-Node multi-process distributed training.

From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""


import sys
import subprocess
import os
import socket 
from argparse import ArgumentParser, REMAINDER


def get_free_port():  
    sock = socket.socket()
    sock.bind(('', 0))
    ip, port = sock.getsockname()
    sock.close()
    return port

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")

    parser.add_argument("--gpus", default="0", type=str,
                        help="CUDA_VISIBLE_DEVICES")
    # Optional arguments for the launch helper
    # parser.add_argument("--nnodes", type=int, default=1,
    #                     help="The number of nodes to use for distributed "
    #                          "training")
    # parser.add_argument("--node_rank", type=int, default=0,
    #                     help="The rank of the node for multi-node distributed "
    #                          "training")
    # parser.add_argument("--nproc_per_node", type=int, default=1,
    #                     help="The number of processes to launch on each node, "
    #                          "for GPU training, this is recommended to be set "
    #                          "to the number of GPUs in your system so that "
    #                          "each process can be bound to a single GPU.")
    # parser.add_argument("--master_addr", default="127.0.0.1", type=str,
    #                     help="Master node (rank 0)'s address, should be either "
    #                          "the IP address or the hostname of node 0, for "
    #                          "single node multi-proc training, the "
    #                          "--master_addr can simply be 127.0.0.1")
    # parser.add_argument("--master_port", default=29500, type=int,
    #                     help="Master node (rank 0)'s free port that needs to "
    #                          "be used for communication during distributed "
    #                          "training")
    # parser.add_argument("--use_env", default=False, action="store_true",
    #                     help="Use environment variable to pass "
    #                          "'local rank'. For legacy reasons, the default value is False. "
    #                          "If set to True, the script will not pass "
    #                          "--local_rank as argument, and will instead set LOCAL_RANK.")
    # parser.add_argument("-m", "--module", default=False, action="store_true",
    #                     help="Changes each process to interpret the launch script "
    #                          "as a python module, executing with the same behavior as"
    #                          "'python -m'.")
    # parser.add_argument("--no_python", default=False, action="store_true",
    #                     help="Do not prepend the training script with \"python\" - just exec "
    #                          "it directly. Useful when the script is not a Python script.")

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

def main():
    args = parse_args()
    
    n_gpus = len(args.gpus.split(","))

    # here, we specify some command line parameters manually,
    # since in Single-Node multi-process distributed training, they often are fixed or computable
    args.nnodes = 1
    args.node_rank = 0
    args.nproc_per_node = n_gpus
    args.master_addr = "127.0.0.1"
    args.master_port = get_free_port()
    args.use_env = False
    args.module = False
    args.no_python = False


    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()

    current_env["CUDA_VISIBLE_DEVICES"] = args.gpus

    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        # print("*****************************************\n"
        #       "Setting OMP_NUM_THREADS environment variable for each process "
        #       "to be {} in default, to avoid your system being overloaded, "
        #       "please further tune the variable for optimal performance in "
        #       "your application as needed. \n"
        #       "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        else:
            if not args.use_env:
                raise ValueError("When using the '--no_python' flag, you must also set the '--use_env' flag.")
            if args.module:
                raise ValueError("Don't use both the '--no_python' flag and the '--module' flag at the same time.")

        cmd.append(args.training_script)

        if not args.use_env:
            cmd.append("--local_rank={}".format(local_rank))

        cmd.extend(args.training_script_args)

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)


if __name__ == "__main__":
    main()
