import torch
import torch.nn as nn
import torch.nn.functional as F

class PWMGenerator(nn.Module):
    def __init__(self, hidden_dim=1280, num_heads=8, pwm_length=35, conv_dim=512, max_asym_id=3, asym_embed_dim=8):
        super().__init__()
        self.pwm_length = pwm_length
        self.asym_embed_dim = asym_embed_dim

        # Chain ID embedding
        self.asym_embed = nn.Embedding(max_asym_id, asym_embed_dim)

        # Conv1D layer
        self.conv1d = nn.Conv1d(in_channels=hidden_dim + asym_embed_dim, out_channels=conv_dim, kernel_size=5, padding=2)
        self.conv_norm = nn.BatchNorm1d(conv_dim)
        self.conv_activation = nn.ReLU()

        self.feature_dropout = nn.Dropout(0.2)

        self.dna_queries = nn.Parameter(torch.randn(1, pwm_length, conv_dim))
        
        # Multi-head attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=conv_dim,
            num_heads=num_heads,
            dropout=0.15,
            batch_first=True
        )

        # Layer normalization for attention input
        self.attn_norm = nn.LayerNorm(conv_dim)
        
        # DBD weight predictor layer
        # self.dbd_weight_predictor = nn.Sequential(
        #     nn.Linear(conv_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )

        # Feature fusion layer
        # self.fusion = nn.Sequential(
        #     nn.Linear(conv_dim, conv_dim),
        #     nn.LayerNorm(conv_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )

        # PWM generator layer
        self.pwm_generator = nn.Sequential(
            nn.Linear(conv_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )

        # 1. Binary DBD classification head (DBD vs. non-DBD).
        # self.dbd_classifier = nn.Sequential(
        #     nn.Linear(conv_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )
        self.dbd_classifier = nn.Sequential(
            nn.Dropout(0.3),            # Discourage reliance on a single feature.
            nn.Linear(conv_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),            # Further reduce overfitting.
            nn.Linear(128, 1)
        )

        # 2. Contrastive-learning projection head for the NT-Xent loss.
        self.projection_head = nn.Sequential(
            nn.Linear(conv_dim, conv_dim),
            nn.ReLU(),
            nn.Linear(conv_dim, 128)
        )

    def forward(self, protein_embeddings, asym_id, protein_masks):
        batch_size = protein_embeddings.size(0)

        # DBD feature mask
        dbd_emb = protein_embeddings * protein_masks.unsqueeze(-1)
        asym_emb = self.asym_embed(asym_id) * protein_masks.unsqueeze(-1)
        combined_emb = torch.cat([dbd_emb, asym_emb], dim=-1)

        # Conv1D extract local features
        conv_input = combined_emb.transpose(1, 2)
        conv_output = self.conv1d(conv_input)
        conv_output = self.conv_norm(conv_output)
        conv_output = self.conv_activation(conv_output)

        conv_output = self.feature_dropout(conv_output)

        conv_output = conv_output.transpose(1, 2)

        dbd_features = conv_output * protein_masks.unsqueeze(-1)

        q = self.dna_queries.expand(batch_size, -1, -1)

        value_features = F.dropout(dbd_features, p=0.3, training=self.training)

        pwm_context, attn_weights = self.cross_attention(
            query=q,
            key=dbd_features,
            value=value_features,
            key_padding_mask=(protein_masks == 0)
        )
        pwm_context = self.attn_norm(pwm_context)

        pwm_logits = self.pwm_generator(pwm_context)
        pwm = F.softmax(pwm_logits, dim=-1)
        # Multi-head attention extract global features
        # attended_features, _ = self.multi_attention(
        #     query=dbd_features,
        #     key=dbd_features,
        #     value=dbd_features,
        #     key_padding_mask=(protein_masks == 0)
        # )

        # attended_features = self.attn_norm(dbd_features + attended_features)
        # attended_features = attended_features * protein_masks.unsqueeze(-1)

        # weight_logits = self.dbd_weight_predictor(dbd_features).squeeze(-1)
        # mask_value = -1e4
        # weight_logits = weight_logits.masked_fill(protein_masks == 0, mask_value)
        # normalized_weights = F.softmax(weight_logits, dim=-1)
        
        # global_features = (dbd_features * normalized_weights.unsqueeze(-1)).sum(dim=1)

        # fused_features = self.fusion(attended_features)
        
        # pwm_logits = self.pwm_generator(fused_features)
        # pwm = pwm_logits.view(-1, self.pwm_length, 4)
        # pwm = F.softmax(pwm, dim=-1)

        residue_importance = attn_weights.mean(dim=1)
        weighted_protein_feat = (dbd_features * residue_importance.unsqueeze(-1)).sum(dim=1)
        # Compute auxiliary-task outputs.
        dbd_logits = self.dbd_classifier(weighted_protein_feat)
        
        # L2-normalized contrastive-learning features.
        projection = self.projection_head(weighted_protein_feat)
        projection = F.normalize(projection, dim=1)
        
        # Return a dictionary for convenient unpacking.
        aux_outputs = {
            "global_features": weighted_protein_feat,
            "projection": projection,
            "dbd_logits": dbd_logits,
            "attn_weights": attn_weights
        }
        # ===========================================

        return pwm, residue_importance, aux_outputs
 
