from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PWMGenerator(nn.Module):
    """
    Ablation options
    ----------------
    "none":
        Full Seq2PWM. Input is an ESM residue embedding tensor with shape
        (B, L, hidden_dim).

    "wo_cnn":
        Uses ESM residue embeddings, but replaces the local Conv1D encoder
        with a residue-wise linear projection.

    "wo_crossattn":
        Uses ESM residue embeddings and the Conv1D encoder, but replaces
        DNA-query cross-attention with masked mean pooling followed by one
        linear layer that predicts all PWM positions.

    "wo_esm":
        Does not use external ESM residue embeddings. Amino-acid token IDs
        are mapped to trainable residue embeddings with nn.Embedding.
        The Conv1D encoder, DNA-query cross-attention, PWM head, DBD head,
        and contrastive-learning head remain unchanged.

    """

    VALID_ABLATIONS = {
        "none",
        "wo_cnn",
        "wo_crossattn",
        "wo_esm",
    }

    def __init__(
        self,
        hidden_dim: int = 1280,
        num_heads: int = 8,
        pwm_length: int = 35,
        conv_dim: int = 512,
        max_asym_id: int = 3,
        asym_embed_dim: int = 8,
        aa_vocab_size: int = 21,
        ablation: str = "none",
    ):
        super().__init__()

        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation: {ablation}. "
                f"Expected one of {sorted(self.VALID_ABLATIONS)}."
            )

        self.ablation = ablation
        self.hidden_dim = hidden_dim
        self.pwm_length = pwm_length
        self.conv_dim = conv_dim
        self.asym_embed_dim = asym_embed_dim

        # Only wo_esm disables external ESM residue embeddings.
        self.use_esm = ablation != "wo_esm"

        # Single-component ablations retain all unrelated components.
        self.use_local_cnn = ablation != "wo_cnn"
        self.use_cross_attention = ablation != "wo_crossattn"

        # ==========================================================
        # Residue input representation
        # ==========================================================
        if not self.use_esm:
            # wo_esm: learn residue representations directly from AA IDs.
            self.aa_embed = nn.Embedding(
                num_embeddings=aa_vocab_size,
                embedding_dim=hidden_dim,
            )

        # ==========================================================
        # Chain identity embedding
        # ==========================================================
        self.asym_embed = nn.Embedding(
            num_embeddings=max_asym_id,
            embedding_dim=asym_embed_dim,
        )

        input_dim = hidden_dim + asym_embed_dim

        # ==========================================================
        # Protein feature encoder
        # ==========================================================
        if self.use_local_cnn:
            # Used by none, wo_crossattn, and wo_esm.
            self.conv1d = nn.Conv1d(
                in_channels=input_dim,
                out_channels=conv_dim,
                kernel_size=5,
                padding=2,
            )
        else:
            # wo_cnn: no communication between neighboring residues.
            self.position_projection = nn.Linear(
                input_dim,
                conv_dim,
            )

        # BatchNorm is retained in every ablation.
        self.feature_norm = nn.BatchNorm1d(conv_dim)
        self.feature_activation = nn.ReLU()

        # ==========================================================
        # Learnable DNA/PWM-position queries
        # ==========================================================
        self.dna_queries = nn.Parameter(
            torch.randn(1, pwm_length, conv_dim)
        )

        # ==========================================================
        # PWM context encoder
        # ==========================================================
        if self.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=conv_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True,
            )
            self.context_norm = nn.LayerNorm(conv_dim)
        else:
            # wo_crossattn:
            # (B, conv_dim) -> (B, pwm_length * 4)
            self.no_attention_pwm_generator = nn.Linear(
                conv_dim,
                pwm_length * 4,
            )

        # ==========================================================
        # PWM prediction head
        # ==========================================================
        self.pwm_generator = nn.Sequential(
            nn.Linear(conv_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )

        # ==========================================================
        # DBD classification head
        # ==========================================================
        self.dbd_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(conv_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # ==========================================================
        # Contrastive-learning projection head
        # ==========================================================
        self.projection_head = nn.Sequential(
            nn.Linear(conv_dim, conv_dim),
            nn.ReLU(),
            nn.Linear(conv_dim, 128),
        )

    @staticmethod
    def masked_mean(
        features: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Uniform masked mean pooling.

        Parameters
        ----------
        features:
            Tensor of shape (B, L, C).
        mask:
            Tensor of shape (B, L), with 1 for a valid residue and
            0 for padding.

        Returns
        -------
        pooled:
            Tensor of shape (B, C).
        residue_weights:
            Tensor of shape (B, L). Valid residues have weight
            1 / number_of_valid_residues.
        """
        float_mask = mask.to(dtype=features.dtype)

        valid_lengths = float_mask.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)

        residue_weights = float_mask / valid_lengths

        pooled = (
            features * residue_weights.unsqueeze(-1)
        ).sum(dim=1)

        return pooled, residue_weights

    def build_residue_embeddings(
        self,
        protein_embeddings: torch.Tensor | None,
        amino_ids: torch.Tensor | None,
        float_mask: torch.Tensor,
    ):
        """
        Build the residue representation used by the downstream encoder.

        For none / wo_cnn / wo_crossattn:
            protein_embeddings must be supplied and have shape
            (B, L, hidden_dim).

        For wo_esm:
            amino_ids must be supplied and have shape (B, L).
            For backward-compatible positional calls, an integer tensor
            supplied through protein_embeddings is also accepted.
        """
        if self.use_esm:
            if protein_embeddings is None:
                raise ValueError(
                    f"ablation='{self.ablation}' requires "
                    "protein_embeddings with shape (B, L, hidden_dim)."
                )

            if protein_embeddings.ndim != 3:
                raise ValueError(
                    "protein_embeddings must have shape "
                    f"(B, L, hidden_dim), got {tuple(protein_embeddings.shape)}."
                )

            if protein_embeddings.size(-1) != self.hidden_dim:
                raise ValueError(
                    "The final dimension of protein_embeddings must equal "
                    f"hidden_dim={self.hidden_dim}, got "
                    f"{protein_embeddings.size(-1)}."
                )

            residue_embeddings = protein_embeddings

        else:
            # Allow either:
            #   model(amino_ids=..., ...)
            # or the positional shorthand:
            #   model(amino_ids_tensor, asym_id, protein_masks)
            if amino_ids is None:
                if (
                    protein_embeddings is not None
                    and protein_embeddings.ndim == 2
                    and protein_embeddings.dtype
                    in (
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.uint8,
                    )
                ):
                    amino_ids = protein_embeddings
                else:
                    raise ValueError(
                        "ablation='wo_esm' requires amino_ids with "
                        "shape (B, L) and an integer dtype."
                    )

            if amino_ids.ndim != 2:
                raise ValueError(
                    "amino_ids must have shape "
                    f"(B, L), got {tuple(amino_ids.shape)}."
                )

            residue_embeddings = self.aa_embed(
                amino_ids.long()
            )

        residue_embeddings = (
            residue_embeddings * float_mask.unsqueeze(-1)
        )

        return residue_embeddings

    def encode_protein(
        self,
        combined_emb: torch.Tensor,
        float_mask: torch.Tensor,
    ):
        if self.use_local_cnn:
            # (B, L, input_dim) -> (B, input_dim, L)
            encoded = self.conv1d(
                combined_emb.transpose(1, 2)
            )

            encoded = self.feature_norm(encoded)
            encoded = self.feature_activation(encoded)

            # (B, conv_dim, L) -> (B, L, conv_dim)
            encoded = encoded.transpose(1, 2)

        else:
            # wo_cnn:
            # residue-wise projection, without neighboring-residue mixing.
            encoded = self.position_projection(combined_emb)

            # BatchNorm1d expects (B, C, L).
            encoded = self.feature_norm(
                encoded.transpose(1, 2)
            ).transpose(1, 2)

            encoded = self.feature_activation(encoded)

        # Remove any non-zero activations at padded positions.
        encoded = encoded * float_mask.unsqueeze(-1)

        return encoded

    def forward(
        self,
        protein_embeddings: torch.Tensor | None = None,
        asym_id: torch.Tensor | None = None,
        protein_masks: torch.Tensor | None = None,
        amino_ids: torch.Tensor | None = None,
    ):
        """
        Recommended calls
        -----------------
        Full model / wo_cnn / wo_crossattn:

            pwm, residue_importance, aux = model(
                protein_embeddings=esm_embeddings,
                asym_id=asym_id,
                protein_masks=protein_masks,
            )

        wo_esm:

            pwm, residue_importance, aux = model(
                amino_ids=amino_ids,
                asym_id=asym_id,
                protein_masks=protein_masks,
            )

        The positional shorthand below is also supported for wo_esm:

            pwm, residue_importance, aux = model(
                amino_ids,
                asym_id,
                protein_masks,
            )

        The learnable PWM-position queries can be obtained from:

            position_queries = aux["position_queries"]

        Their shape is (B, pwm_length, conv_dim). Because they are expanded
        from one shared nn.Parameter, all samples in a batch initially see
        the same position-query tensor.
        """
        if asym_id is None or protein_masks is None:
            raise ValueError(
                "Both asym_id and protein_masks must be provided."
            )

        if protein_masks.ndim != 2:
            raise ValueError(
                "protein_masks must have shape "
                f"(B, L), got {tuple(protein_masks.shape)}."
            )

        if asym_id.shape != protein_masks.shape:
            raise ValueError(
                "asym_id and protein_masks must have the same shape, "
                f"got {tuple(asym_id.shape)} and "
                f"{tuple(protein_masks.shape)}."
            )

        batch_size = protein_masks.size(0)

        # MultiheadAttention cannot produce meaningful output for a
        # completely padded sequence.
        if self.use_cross_attention and (protein_masks.sum(dim=1) == 0).any():
            raise ValueError(
                "Every sample must contain at least one valid residue "
                "when cross-attention is enabled."
            )

        # The mask dtype should follow the residue representation dtype.
        if self.use_esm:
            if protein_embeddings is None:
                # build_residue_embeddings will raise a more detailed error.
                mask_dtype = torch.float32
            else:
                mask_dtype = protein_embeddings.dtype
        else:
            mask_dtype = self.aa_embed.weight.dtype

        float_mask = protein_masks.to(dtype=mask_dtype)
        bool_padding_mask = protein_masks == 0

        # ==========================================================
        # Input feature fusion
        # ==========================================================
        residue_embeddings = self.build_residue_embeddings(
            protein_embeddings=protein_embeddings,
            amino_ids=amino_ids,
            float_mask=float_mask,
        )

        # If asym_id uses a negative padding value in the dataset,
        # replace it with a valid index before calling the model.
        asym_emb = self.asym_embed(asym_id.long())
        asym_emb = asym_emb * float_mask.unsqueeze(-1)

        combined_emb = torch.cat(
            [residue_embeddings, asym_emb],
            dim=-1,
        )

        # ==========================================================
        # Residue-level protein encoding
        # ==========================================================
        dbd_features = self.encode_protein(
            combined_emb=combined_emb,
            float_mask=float_mask,
        )

        # ==========================================================
        # Learnable PWM-position representations
        # ==========================================================
        # This tensor is returned through aux_outputs for visualization,
        # diagnostics, or additional regularization.
        #
        # Shape: (B, pwm_length, conv_dim)
        position_queries = self.dna_queries.expand(
            batch_size,
            -1,
            -1,
        )

        # ==========================================================
        # PWM context generation
        # ==========================================================
        if self.use_cross_attention:
            pwm_context, attn_weights = self.cross_attention(
                query=position_queries,
                key=dbd_features,
                value=dbd_features,
                key_padding_mask=bool_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )

            # All cross-attention variants use the attention output directly,
            # without adding the learnable position query as a residual.
            pwm_context = self.context_norm(pwm_context)

            # attn_weights shape:
            # (B, pwm_length, protein_length)
            residue_importance = attn_weights.mean(dim=1)

            weighted_protein_feat = (
                dbd_features
                * residue_importance.unsqueeze(-1)
            ).sum(dim=1)

            pwm_logits = self.pwm_generator(pwm_context)

        else:
            # ======================================================
            # wo_crossattn
            # ======================================================
            weighted_protein_feat, residue_importance = (
                self.masked_mean(
                    features=dbd_features,
                    mask=protein_masks,
                )
            )

            pwm_logits = self.no_attention_pwm_generator(
                weighted_protein_feat
            ).view(
                batch_size,
                self.pwm_length,
                4,
            )

            # Compatibility with downstream code:
            # these are uniform pooling weights, not learned attention.
            attn_weights = (
                residue_importance
                .unsqueeze(1)
                .expand(-1, self.pwm_length, -1)
            )

        # ==========================================================
        # PWM prediction
        # ==========================================================
        pwm = F.softmax(pwm_logits, dim=-1)

        # ==========================================================
        # Auxiliary tasks
        # ==========================================================
        dbd_logits = self.dbd_classifier(
            weighted_protein_feat
        )

        projection = self.projection_head(
            weighted_protein_feat
        )
        projection = F.normalize(
            projection,
            p=2,
            dim=1,
        )

        aux_outputs = {
            "global_features": weighted_protein_feat,
            "projection": projection,
            "dbd_logits": dbd_logits,
            "attn_weights": attn_weights,
            "position_queries": position_queries,
        }

        return pwm, residue_importance, aux_outputs
