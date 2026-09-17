from typing import Literal


class Config:
    message: str = "SFTGNN graph-only model"
    model_name: Literal[
        "SFTGNN",
        "SFTGNNMultimodal",
        "TextResidualSHFMat",
        "AdaptiveGraphPreservingTextResidualSHFMat",
    ] = "SFTGNN"
    random_seed: int = 123
    run_name: str = ""
    validation_only: bool = False

    node_feature: int = 128
    num_layers: int = 5
    dist_dim: int = 64
    sh_dim: int = 64
    use_sh: bool = True
    use_dist: bool = True

    lr: float = 1e-3
    max_lr: float = 1e-3
    weight_decay: float = 1e-4
    begin_epoch: int = 0
    num_epoch: int = 300

    latent_dim: int = 128
    text_emb_dim: int = 768
    text_hidden_dim: int = 128
    text_out_dim: int = 64
    use_contrastive: bool = False
    contrastive_temperature: float = 0.07
    contrastive_loss_weight: float = 0.1
    property_fusion: Literal["crystal_only", "concat"] = "concat"
    text_emb_path: str = ""

    text_residual_hidden_dim: int = 128
    text_residual_dropout: float = 0.1
    text_residual_alpha_init: float = 0.03
    text_residual_alpha_max: float = 0.2
    text_residual_lambda_delta: float = 0.0

    adaptive_gate_hidden_dim: int = 128
    adaptive_alpha_init: float = 0.02
    adaptive_alpha_max: float = 0.1
    adaptive_alpha_floor: float = 0.0
    adaptive_correction_mode: Literal["learned", "fixed", "floor"] = "learned"
    adaptive_lambda_graph: float = 0.5
    adaptive_lambda_delta: float = 0.001
    adaptive_lambda_residual_fit: float = 0.0
    adaptive_text_dropout: float = 0.2

    @staticmethod
    def info() -> str:
        attrs = [
            "model_name", "random_seed", "node_feature", "num_layers",
            "dist_dim", "sh_dim", "use_sh", "lr", "max_lr",
            "weight_decay",
        ]
        if Config.model_name != "SFTGNN":
            attrs += [
                "text_emb_dim", "text_hidden_dim", "text_out_dim",
                "property_fusion", "use_contrastive",
            ]
        if Config.model_name == "TextResidualSHFMat":
            attrs += [
                "text_residual_hidden_dim", "text_residual_dropout",
                "text_residual_alpha_init", "text_residual_alpha_max",
                "text_residual_lambda_delta",
            ]
        if Config.model_name == "AdaptiveGraphPreservingTextResidualSHFMat":
            attrs += [
                "adaptive_gate_hidden_dim", "adaptive_alpha_init",
                "adaptive_alpha_max", "adaptive_alpha_floor",
                "adaptive_correction_mode", "adaptive_lambda_graph",
                "adaptive_lambda_delta", "adaptive_lambda_residual_fit",
                "adaptive_text_dropout",
            ]
        lines = ["model_config:", "{"]
        lines.extend(f"    {attr}={getattr(Config, attr)}" for attr in attrs)
        lines.append("}")
        return "\n".join(lines)
