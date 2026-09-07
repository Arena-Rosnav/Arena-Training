"""Argument parsing for arena_training scripts."""

import argparse
from collections.abc import Callable


def new_training_args(parser: argparse.ArgumentParser):
    """Program arguments for the training script."""
    parser.add_argument(
        "--config",
        type=str,
        metavar="[config name]",
        default="sb_training_config.yaml",
        help="name of the config file",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="robot model name (e.g. 'jackal', 'burger'); overrides robot_model in config",
    )


def parse_training_args(
    args: list[str] | None = None,
    ignore_unknown: bool = True,
) -> tuple[argparse.Namespace, list[str]]:
    """Parser for the training script."""
    return parse_various_args(args, [new_training_args], [], ignore_unknown)


def parse_various_args(
    args: list[str] | None,
    arg_populate_funcs: list[Callable[[argparse.ArgumentParser], None]],
    arg_check_funcs: list[Callable[[argparse.Namespace], None]],
    ignore_unknown: bool,
) -> tuple[argparse.Namespace, list[str]]:
    """Generic arg parsing function."""
    parser = argparse.ArgumentParser()

    for func in arg_populate_funcs:
        func(parser)

    if ignore_unknown:
        parsed_args, unknown_args = parser.parse_known_args(args=args)
    else:
        parsed_args = parser.parse_args(args=args)
        unknown_args = []

    for func in arg_check_funcs:
        func(parsed_args)

    print_args(parsed_args)
    return parsed_args, unknown_args


def print_args(args: argparse.Namespace):
    print("\n-------------------------------")
    print("            ARGUMENTS          ")
    for k in args.__dict__:
        print(f"- {k} : {args.__dict__[k]}")
    print("--------------------------------\n")
