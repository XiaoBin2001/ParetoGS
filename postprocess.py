"""
Post-processing command-line entry point.

It rebuilds the dataset and trained Gaussian model from `model_path`, then
delegates rendering, metric computation, FPS benchmarking, and summary saving to
`utils.evaluation.render_metric`.
"""
from argparse import ArgumentParser

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args, parse_bool_arg
from utils.general_utils import safe_state


def process_splits(
    dataset: GroupParams,
    iteration: int,
    pipeline: GroupParams,
    include_train: bool,
    skip_test: bool,
    no_save_test_images: bool,
    no_save_train_images: bool,
    fps_warmup_rounds: int,
    fps_measure_rounds: int,
    prefix_budgets: list[int],
    prefix_adapter_path: str,
) -> None:
    """
        Import and run the heavy post-processing implementation on demand.
    """
    from utils.evaluation.render_metric import process_splits as run_process_splits

    run_process_splits(
        dataset,
        iteration,
        pipeline,
        include_train,
        skip_test,
        no_save_test_images,
        no_save_train_images,
        fps_warmup_rounds,
        fps_measure_rounds,
        prefix_budgets,
        prefix_adapter_path,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Post-processing script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--include_train", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--skip_test", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--no_save_test_images", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--no_save_train_images", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--fps_warmup_rounds", default=20, type=int)
    parser.add_argument("--fps_measure_rounds", default=20, type=int)
    parser.add_argument("--prefix_budgets", nargs="+", type=int, default=[])
    parser.add_argument("--prefix_adapter_path", default="", type=str)
    parser.add_argument("--quiet", default=False, nargs="?", const=True, type=parse_bool_arg)
    args = get_combined_args(parser)
    print("Post-processing " + args.model_path)

    safe_state(args.quiet)
    # Keep argument extraction here so the evaluation module receives the same grouped objects as training.
    dataset_args = model.extract(args)
    try:
        process_splits(
            dataset=dataset_args,
            iteration=args.iteration,
            pipeline=pipeline.extract(args),
            include_train=args.include_train,
            skip_test=args.skip_test,
            no_save_test_images=args.no_save_test_images,
            no_save_train_images=args.no_save_train_images,
            fps_warmup_rounds=args.fps_warmup_rounds,
            fps_measure_rounds=args.fps_measure_rounds,
            prefix_budgets=args.prefix_budgets,
            prefix_adapter_path=args.prefix_adapter_path,
        )
    except KeyboardInterrupt:
        print("\nPost-processing interrupted by user.")
        raise SystemExit(130)
