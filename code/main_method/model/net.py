import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from .sublayer import SFTConv, RBFExpansion
from .config import Config as model_config
from data.config import Config as data_config


class SFTGNN(nn.Module):
    """
        SFTGNN Standard Model
    """

    def __init__(self, num_layers: int = 5,
                 in_dim: int = 92,
                 node_feature: int = 128,
                 dist_dim: int = 64,
                 sh_dim: int = 64,
                 use_sh: bool = True,
                 use_dist: bool = True,
                 sh_in_dim: int | None = None):
        """
            SFTGNN Standard Model
        :param num_layers:
        :param in_dim: The feature dimension of the AtomFeatureType used
        :param node_feature:
        :param dist_dim:
        :param sh_dim:
        :param use_sh:
        :param use_dist:
        """
        super().__init__()
        if in_dim != 1:  # AtomFeatureType = CGCNN or CrysAtomVec
            self.atom_embedding = nn.Linear(in_dim, node_feature)
        else:  # AtomFeatureType = AtomicNumber
            self.atom_embedding = nn.Sequential(
                nn.Embedding(94, node_feature),
                nn.Linear(node_feature, node_feature)
            )
        self.num_layers = num_layers
        self.use_sh = use_sh
        self.use_dist = use_dist
        self.layers = nn.ModuleList(
            [SFTConv(node_feature, dist_dim, sh_dim, use_sh, use_dist) for _ in range(self.num_layers)])
        sh_in_dim = data_config.spherical_harmonics_l ** 2 if sh_in_dim is None else int(sh_in_dim)
        self.sh_mlp = nn.Sequential(
            nn.Linear(sh_in_dim, 32),
            nn.SiLU(),
            nn.Linear(32, sh_dim),
        )
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0.125, vmax=1.4, bins=64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, dist_dim),
        )
        self.fc = nn.Sequential(
            nn.Linear(node_feature, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, data) -> torch.Tensor:
        node_features = self.atom_embedding(data.x)
        index = data.edge_index
        sh = self.sh_mlp(data.sh)
        with torch.no_grad():
            dist = 1 / data.edge_attr.unsqueeze(-1)
        dist = self.rbf(dist)
        edge_attr = torch.cat([dist, sh], dim=-1)
        for layer in self.layers:
            node_features = layer(node_features, index, edge_attr)
        features = scatter(node_features, data.batch, dim=0, reduce="mean")
        out = self.fc(features)
        return out.squeeze()

    def forward_embedding(self, data) -> torch.Tensor:
        """
        返回图级表示 h_crystal [B, node_feature]，不经过 property head。
        多模态时用于投影到 latent_dim 作为 embeddings['crystal']。
        """
        node_features = self.atom_embedding(data.x)
        index = data.edge_index
        sh = self.sh_mlp(data.sh)
        with torch.no_grad():
            dist = 1 / data.edge_attr.unsqueeze(-1)
        dist = self.rbf(dist)
        edge_attr = torch.cat([dist, sh], dim=-1)
        for layer in self.layers:
            node_features = layer(node_features, index, edge_attr)
        h_crystal = scatter(node_features, data.batch, dim=0, reduce="mean")
        return h_crystal


class TextMLP(nn.Module):
    """
    文本嵌入投影（按论文/描述公式）：
    Z_T = W2(ReLU(W1·Z_CLS))
      - W1 ∈ R^{768×128}
      - W2 ∈ R^{128×64}
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SFTGNNMultimodal(nn.Module):
    """
    多模态晶体性能预测：拼接融合（Concatenation）。

    - 晶体编码器：SFTGNN.forward_embedding -> 线性投影 -> emb_crystal [B, crystal_latent_dim]（默认 128 维）
    - 文本编码器：batch.text_emb (768) -> TextMLP -> emb_text [B, text_out_dim]（默认 64 维）
    - 融合方式：feature = concat([emb_crystal, emb_text])，再送入全连接网络预测物理属性。
    """

    def __init__(
        self,
        sftgnn: SFTGNN,
        crystal_dim: int = 128,
        latent_dim: int = 128,
        text_in_dim: int = 768,
        text_hidden_dim: int = 128,
        text_out_dim: int = 64,
        property_fusion: str = "concat",
    ):
        super().__init__()
        self.sftgnn = sftgnn
        self.crystal_latent_dim = latent_dim
        self.text_out_dim = text_out_dim
        self.property_fusion = property_fusion

        if crystal_dim != latent_dim:
            self.crystal_proj = nn.Linear(crystal_dim, latent_dim)
        else:
            self.crystal_proj = nn.Identity()

        self.text_mlp = TextMLP(
            in_dim=text_in_dim, hidden_dim=text_hidden_dim, out_dim=text_out_dim
        )

        if property_fusion == "crystal_only":
            self.property_head = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )
        else:
            fusion_in_dim = latent_dim + text_out_dim
            self.property_head = nn.Sequential(
                nn.Linear(fusion_in_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 1),
            )

    def forward(self, data) -> dict:
        """
        data 需包含图数据及 batch.text_emb [B, 768]（多模态时）。
        Returns:
            embeddings: dict with 'crystal' [B, crystal_latent_dim], 'text' [B, text_out_dim]
            property_pred: [B] 属性预测值
        """
        h_crystal = self.sftgnn.forward_embedding(data)
        emb_crystal = self.crystal_proj(h_crystal)
        B = emb_crystal.size(0)

        if hasattr(data, "text_emb") and data.text_emb is not None:
            # PyG 默认把 graph-level 的 text_emb 沿 dim=0 拼成 (B*768,)；需 reshape 为 (B, 768)
            text_in = data.text_emb
            if text_in.dim() == 1:
                text_in = text_in.view(B, -1)
            emb_text = self.text_mlp(text_in)
        else:
            emb_text = torch.zeros(B, self.text_out_dim, device=emb_crystal.device, dtype=emb_crystal.dtype)

        embeddings = {"crystal": emb_crystal, "text": emb_text}

        if self.property_fusion == "crystal_only":
            prop_in = emb_crystal
        else:
            prop_in = torch.cat([emb_crystal, emb_text], dim=-1)
        property_pred = self.property_head(prop_in).squeeze(-1)

        return {"embeddings": embeddings, "property_pred": property_pred}

    def forward_contrastive_logits(self, embeddings: dict, temperature: float = 0.07) -> torch.Tensor:
        """
        返回 logits = emb_text @ emb_crystal.T / temperature [B, B]，供评估/检索用。
        训练时对比损失请用 src.loss.infonce.infonce_loss(emb_text, emb_crystal, temperature)。
        """
        if embeddings["crystal"].size(-1) != embeddings["text"].size(-1):
            raise ValueError(
                "Contrastive logits require same embedding dim, but got "
                f"crystal={embeddings['crystal'].size(-1)} and text={embeddings['text'].size(-1)}."
            )
        c = F.normalize(embeddings["crystal"], dim=-1)
        t = F.normalize(embeddings["text"], dim=-1)
        logits = (t @ c.t()) / temperature
        return logits
