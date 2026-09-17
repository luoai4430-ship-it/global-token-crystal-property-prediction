import argparse
import os

from data.config import Config as data_config
from data.config import JarvisTarget, MPTarget
from model.config import Config as model_config
from train_and_test import TrainConfigManager, test_with_pretrained_model, train_and_test


def build_parser() -> argparse.ArgumentParser:
    targets = [f"Jarvis-{t.name}" for t in JarvisTarget] + [f"MP-{t.name}" for t in MPTarget]
    parser = argparse.ArgumentParser(description="Train the paper's graph-only, concat, or residual model.")
    parser.add_argument("--task", choices=["train_and_test", "test"], required=True)
    parser.add_argument("--target", choices=targets, required=True)
    parser.add_argument(
        "--model",
        choices=["graphonly", "concat", "textresidual", "adaptive_residual"],
        required=True,
    )
    parser.add_argument("--use_cif_csv", action="store_true")
    parser.add_argument("--cif_dir")
    parser.add_argument("--text_emb_path")
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha_init", type=float, default=0.03)
    parser.add_argument("--alpha_max", type=float, default=0.20)
    parser.add_argument("--alpha_floor", type=float, default=0.0)
    parser.add_argument(
        "--correction_mode",
        choices=["learned", "fixed", "floor"],
        default="learned",
    )
    parser.add_argument("--lambda_delta", type=float, default=0.0)
    parser.add_argument("--lambda_graph", type=float, default=0.5)
    parser.add_argument("--lambda_residual_fit", type=float, default=0.0)
    parser.add_argument("--text_dropout", type=float, default=0.2)
    parser.add_argument("--gate_hidden_dim", type=int, default=128)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--validation_only", action="store_true")
    return parser


def configure(args: argparse.Namespace) -> None:
    dataset, target = args.target.split("-", 1)
    data_config.target = JarvisTarget[target] if dataset == "Jarvis" else MPTarget[target]
    data_config.batch_size = args.batch_size
    data_config.use_cif_csv = args.use_cif_csv
    data_config.cif_dir = args.cif_dir or ""

    if args.use_cif_csv:
        if not args.cif_dir or not os.path.isdir(args.cif_dir):
            raise ValueError("--use_cif_csv requires an existing --cif_dir")

    model_config.num_epoch = args.num_epoch
    model_config.random_seed = args.seed
    model_config.lr = args.lr
    model_config.max_lr = args.max_lr
    model_config.weight_decay = args.weight_decay
    model_config.run_name = args.run_name
    model_config.validation_only = args.validation_only

    if args.model == "graphonly":
        model_config.model_name = "SFTGNN"
        model_config.message = "SFTGNN graph-only model"
        data_config.use_multimodal = False
        return

    if not args.text_emb_path or not os.path.isfile(args.text_emb_path):
        raise ValueError("concat and textresidual require an existing --text_emb_path")
    data_config.use_multimodal = True
    data_config.text_emb_path = args.text_emb_path
    model_config.text_emb_path = args.text_emb_path

    if args.model == "concat":
        model_config.model_name = "SFTGNNMultimodal"
        model_config.property_fusion = "concat"
        model_config.use_contrastive = False
        model_config.message = "Direct graph-text concatenation"
    elif args.model == "textresidual":
        model_config.model_name = "TextResidualSHFMat"
        model_config.text_residual_alpha_init = args.alpha_init
        model_config.text_residual_alpha_max = args.alpha_max
        model_config.text_residual_lambda_delta = args.lambda_delta
        model_config.message = "Graph-anchored bounded-gain textual residual"
    else:
        model_config.model_name = "AdaptiveGraphPreservingTextResidualSHFMat"
        model_config.adaptive_alpha_init = args.alpha_init
        model_config.adaptive_alpha_max = args.alpha_max
        model_config.adaptive_alpha_floor = args.alpha_floor
        model_config.adaptive_correction_mode = args.correction_mode
        model_config.adaptive_lambda_graph = args.lambda_graph
        model_config.adaptive_lambda_delta = args.lambda_delta
        model_config.adaptive_lambda_residual_fit = args.lambda_residual_fit
        model_config.adaptive_text_dropout = args.text_dropout
        model_config.adaptive_gate_hidden_dim = args.gate_hidden_dim
        model_config.message = (
            "Residual-fit graph-preserving textual correction"
            if args.lambda_residual_fit > 0
            else "Adaptive graph-preserving textual residual"
        )


def main() -> None:
    args = build_parser().parse_args()
    configure(args)
    with TrainConfigManager(None):
        if args.task == "train_and_test":
            train_and_test()
        else:
            test_with_pretrained_model()


if __name__ == "__main__":
    main()
