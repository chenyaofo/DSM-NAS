import sys
import json
import os
import typing
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.optim as optim

from codebase.engine.accuracy_predictor import train_acc_pred_complete
from codebase.database.nasbench201 import NASBench201DataBase
from codebase.core.dynamic_graph import DynamicGraph
from codebase.core.arch_representation.nasbench201 import NASBench201Architecture
from codebase.controller.nasbench201 import NASBench201ModificationController, NASBench201ModificationController_DifferentPolicy, NASBench201ModificationController_SelectBestSpace
# from codebase.third_party.ofa.evaluate_archs import OFAArchitectureEvaluator

from codebase.torchutils.metrics import MovingAverageMetric
from codebase.torchutils import logger, summary_writer, output_directory
from codebase.torchutils.common import set_reproducible
from codebase.torchutils.common import torchsave, auto_device
from codebase.torchutils.typed_args import TypedArgs, add_argument

import copy



@dataclass
class Args(TypedArgs):
    seed: int = add_argument('--seed', default=23555)

    n_trial: int = add_argument('--n_trial', default=20)

    time_limit: float = add_argument('--time_limit',  default=25000)
    dataset: str = add_argument("--dataset", default="cifar10")
    search_space: str = add_argument("--search_space", default="ofa")

    K: int = add_argument('-K',  default=3)
    n_steps: int = add_argument('--n_steps',  default=4)
    max_iterations: int = add_argument('--max_iterations',  default=200)

    hidden_state: int = add_argument('--hidden_state',  default=64)

    # controller_learning_rate: float = add_argument('--controller_learning_rate', default=0.0095)
    # controller_weight_decay: float = add_argument('--controller_weight_decay', default=2.37e-05)
    # entropy_coeff: float = add_argument('--entropy_coeff', default=0.0005)

    controller_learning_rate: float = add_argument('--controller_learning_rate', default=0.009446763714621489)
    controller_weight_decay: float = add_argument('--controller_weight_decay', default=2.3740059139364297e-05)
    entropy_coeff: float = add_argument('--entropy_coeff', default=0.0005049559768158195)

    madds: float = add_argument('--madds', default=600.0)

    imagenet_root: str = add_argument('--imagenet_root', default="/home/chenyaofo/datasets/imagenet/")
    use_resolution: bool = add_argument("--use_resolution", action="store_true")
    use_our_supernet: bool = add_argument("--use_our_supernet", action="store_true")
    no_edge: bool = add_argument("--no_edge", action="store_true")
    no_evaluate: bool = add_argument("--no_evaluate", action="store_true")
    transfer: bool = add_argument("--transfer", action="store_true")
    use_absolute_perf: bool = add_argument("--use_absolute_perf", action="store_true")
    different_policy: bool = add_argument("--different_policy", action="store_true")

    arch_data_root: str = add_argument('--arch_data_root', default="assets/NAS-Bench-201-iepoch12.json")
    drop_rate: typing.Optional[float] = add_argument('--drop_rate', default=0.3)
    huber_beta: typing.Optional[float] = add_argument('--huber_beta', default=None)
    update_freq: int = add_argument('--update_freq',  default=1)
    select_best_space: bool = add_argument("--select_best_space", action="store_true")


args, _ = Args.from_known_args(sys.argv)

database = NASBench201DataBase.from_file(args.arch_data_root, args.dataset)


def search(args: Args, init_archs, database, dataset):
    iter_ = 0
    while len(init_archs) != args.K:
        arch = NASBench201Architecture.random()
        init_archs.append(arch)
    for arch in init_archs:
        arch.obtain_acc_by(database)
        arch.top1_acc = arch.metadata[dataset]['val_acc']
    for arch in init_archs:
        logger.info(f"Init Arch={arch.to_string()}, val acc={arch.metadata[dataset]['val_acc']:.2f}%")
    dynamic_graph = DynamicGraph(init_archs)
    if args.different_policy:
        controller = NASBench201ModificationController_DifferentPolicy(
            K=args.K, n_steps=args.n_steps, device=auto_device, in_features=37,
            no_edge=args.no_edge).to(device=auto_device)
    elif args.select_best_space:
        controller = NASBench201ModificationController_SelectBestSpace(
            K=args.K, n_steps=args.n_steps, device=auto_device, in_features=37,
            no_edge=args.no_edge, dataset=args.dataset).to(device=auto_device)
    else:
        controller = NASBench201ModificationController(
            K=args.K, n_steps=args.n_steps, device=auto_device, in_features=37,
            no_edge=args.no_edge).to(device=auto_device)
    # controller = Controller(K=args.K, n_steps=args.n_steps, device=auto_device, in_features=37,
    #                         no_edge=args.no_edge).to(device=auto_device)
    optimizer = optim.Adam(controller.parameters(), lr=args.controller_learning_rate,
                           weight_decay=args.controller_weight_decay)
    baseline = MovingAverageMetric(gamma=0.9)

    time_cost = 0.0
    traverse_archs = []
    update_times = 0
    while True:
        iter_ += 1
        edit_seq, logp, entropy, max_entropy = controller(dynamic_graph)
        start_arch_index, *edit = edit_seq
        start_arch = dynamic_graph.at(start_arch_index)
        end_arch = start_arch.apply(edit)
        end_arch.obtain_acc_by(database)
        end_arch.top1_acc = end_arch.metadata[dataset]['val_acc']
        time_cost += end_arch.metadata[dataset]['time']

        traverse_archs.append(end_arch)

        to_update_index = start_arch_index
        if iter_ % args.update_freq == 0:
            update_times += dynamic_graph.update(start_arch_index, end_arch, iter_)

        # update the parameters of the controller
        if args.use_absolute_perf:
            perf_improvement = end_arch.metadata[dataset]['val_acc']
        else:
            perf_improvement = (end_arch.metadata[dataset]['val_acc']-start_arch.metadata[dataset]['val_acc'])
        baseline.update(perf_improvement)
        reward = perf_improvement-baseline.value
        policy_loss = -logp*reward-args.entropy_coeff*entropy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        diversity_mean, diversity_std = dynamic_graph.diversity

        if iter_ % 10 == 0 or time_cost > args.time_limit or iter_ > args.max_iterations:
            logger.info(", ".join([
                f"Iteration={iter_:05d}/{args.max_iterations:05d}",
                f"time cost={time_cost:.2f}",
                f"select sub-space={edit_seq[0]}",
                f"perf_improvement={perf_improvement:.4f}",
                f"reward={reward:.4f}",
                f"logp={logp.item():.4f}",
                f"entropy={entropy.item():.4f} ({entropy.item()/max_entropy:.4f})",
                f"diversity={diversity_mean:.2f}±{diversity_std:.2f}",
                f"DynamicGraph min={dynamic_graph.min_accuracy:.2f}%",
                f"DynamicGraph max={dynamic_graph.max_accuracy:.2f}%",
                f"DynamicGraph mean={dynamic_graph.mean_accuracy:.2f}%",
                f"Last update dynamic graph iteration is {dynamic_graph.last_update_iter_:05d}"
            ]))

            summary_writer.add_scalar("search/logp", logp.item(), iter_)
            summary_writer.add_scalar("search/perf_improvement", perf_improvement, iter_)
            summary_writer.add_scalar("search/reward", reward, iter_)
            summary_writer.add_scalar("search/entropy", entropy.item(), iter_)
            summary_writer.add_scalar("search/entropy_percent", entropy.item()/max_entropy, iter_)
            summary_writer.add_scalar("search/diversity_mean", diversity_mean, iter_)
            summary_writer.add_scalar("search/diversity_std", diversity_std, iter_)
            summary_writer.add_scalar("search/min_accuracy", dynamic_graph.min_accuracy, iter_)
            summary_writer.add_scalar("search/max_accuracy", dynamic_graph.max_accuracy, iter_)
            summary_writer.add_scalar("search/mean_accuracy", dynamic_graph.mean_accuracy, iter_)

        if time_cost > args.time_limit or iter_ > args.max_iterations:
            return dynamic_graph, traverse_archs, update_times


def analyze_graph(dynamic_graph: DynamicGraph, dataset: str):
    best_arch = None
    best_acc = 0.0
    for i, node in enumerate(dynamic_graph.nodes):
        if node.metadata[dataset]['val_acc'] > best_acc:
            best_acc = node.metadata[dataset]['val_acc']
            best_arch = node
    return (best_arch.metadata["cifar10"]['test_acc'],
            best_arch.metadata["cifar100"]['test_acc'],
            best_arch.metadata["imagenet"]['test_acc'])


def analyze_traverse_archs(traverse_archs, dataset: str):
    best_arch = None
    best_acc = 0.0
    for i, node in enumerate(traverse_archs):
        if node.metadata[dataset]['val_acc'] > best_acc:
            best_acc = node.metadata[dataset]['val_acc']
            best_arch = node
    return (best_arch.metadata["cifar10"]['test_acc'],
            best_arch.metadata["cifar100"]['test_acc'],
            best_arch.metadata["imagenet"]['test_acc'])


def main(_args, database):
    logger.info(_args)
    with open(os.path.join(output_directory, "results.txt"), "w") as f:
        results = []
        c100_results = []
        imagenet_results = []
        traverse_results = []
        update_times_list = []
        for i in range(_args.n_trial):
            set_reproducible(_args.seed+i)
            graph, traverse_archs, update_times = search(_args, list(), database, _args.dataset)
            with open(os.path.join(output_directory, f"graph-{i}.txt"), "w") as f1:
                for _, node in enumerate(graph.nodes):
                    f1.write(node.to_string()+"\n")
            results.append(analyze_graph(graph, _args.dataset))
            traverse_results.append(analyze_traverse_archs(traverse_archs, _args.dataset))
            update_times_list.append(update_times)

            if _args.transfer:
                _args.time_limit = 100000000
                c100_graph, *_ = search(_args, graph.nodes, database, "cifar100")
                c100_results.append(analyze_graph(c100_graph, "cifar100"))

                imagenet_graph, *_ = search(_args, graph.nodes, database, "imagenet")
                imagenet_results.append(analyze_graph(imagenet_graph, "imagenet"))

        results = torch.tensor(results)
        avg = torch.mean(results, dim=0).tolist()
        std = torch.std(results, dim=0).tolist()
        print(f"Original Search", file=f)
        print(avg, file=f)
        print(std, file=f)
        str_result = f"CIFAR10 {avg[0]:.2f}±{std[0]:.2f}%, CIFAR100 {avg[1]:.2f}±{std[1]:.2f}%, ImageNet {avg[2]:.2f}±{std[2]:.2f}%"
        logger.info(str_result)
        with open(os.path.join(output_directory, "original.txt"), "w") as f1:
            if _args.dataset == "cifar10":
                rev = avg[0]
                perfs = results[:, 0].tolist()
            elif _args.dataset == "cifar100":
                rev = avg[1]
                perfs = results[:, 1].tolist()
            elif _args.dataset == "imagenet":
                rev = avg[2]
                perfs = results[:, 2].tolist()
            print(f"{perfs}", file=f1)

        traverse_results = torch.tensor(traverse_results)
        avg = torch.mean(traverse_results, dim=0).tolist()
        std = torch.std(traverse_results, dim=0).tolist()
        print(f"traverse_results", file=f)
        print(avg, file=f)
        print(std, file=f)
        # logger.info(f"traverse_results CIFAR10 {avg[0]:.2f}±{std[0]:.2f}%, CIFAR100 {avg[1]:.2f}±{std[1]:.2f}%, ImageNet {avg[2]:.2f}±{std[2]:.2f}%,")

        # logger.info(f"avg update times {sum(update_times_list)/len(update_times_list)}")

        f.flush()

        if _args.transfer:
            c100_results = torch.tensor(c100_results)
            print(f"Transfer to CIFAR100 Search", file=f)
            print(torch.mean(c100_results, dim=0).tolist(), file=f)
            print(torch.std(c100_results, dim=0).tolist(), file=f)
            print("-"*50, file=f)
            with open(os.path.join(output_directory, "tranfer_cifar100.txt"), "w") as f1:
                perfs = c100_results[:, 1].tolist()
                print(f"{perfs}", file=f1)
            imagenet_results = torch.tensor(imagenet_results)
            print(f"Transfer to Imagenet Search", file=f)
            print(torch.mean(imagenet_results, dim=0).tolist(), file=f)
            print(torch.std(imagenet_results, dim=0).tolist(), file=f)
            print("-"*50, file=f)
            with open(os.path.join(output_directory, "tranfer_imagenet.txt"), "w") as f1:
                perfs = imagenet_results[:, 2].tolist()
                print(f"{perfs}", file=f1)
        f.flush()
        return rev, str_result


if __name__ == "__main__":
    main(args, database)
