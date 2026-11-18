import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            return x.float()
        weights = torch.softmax(self.attention(x.float()).squeeze(-1), dim=1)
        return torch.sum(x.float() * weights.unsqueeze(-1), dim=1)

# 1. Early Fusion (LFDNN)
class LFDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_net = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.2))
        self.audio_net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Dropout(0.2))
        self.vision_net = nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Dropout(0.2))
        self.fusion = nn.Sequential(nn.Linear(272, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, text, audio, vision):
        combined = torch.cat([
            self.text_net(text.mean(dim=1)),
            self.audio_net(audio.mean(dim=1)),
            self.vision_net(vision.mean(dim=1))
        ], dim=1)
        return self.fusion(combined)

# 2. Late Fusion (LF-DNN)
class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear_2(F.relu(self.linear_1(self.drop(self.norm(x)))))

class TextSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, output_size, dropout):
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        return self.linear_1(self.dropout(hidden[-1]))

class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_subnet = SubNet(5, 16, 0.1)
        self.video_subnet = SubNet(20, 16, 0.1)
        self.text_subnet = TextSubNet(768, 128, 64, 0.1)
        self.post_fusion = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, text, audio, vision):
        audio_h = self.audio_subnet(audio.mean(dim=1))
        vision_h = self.video_subnet(vision.mean(dim=1))
        text_h = self.text_subnet(text)
        return self.post_fusion(torch.cat([audio_h, vision_h, text_h], dim=-1))

# 3. Hybrid Fusion (CENET)
class CrossModalEnhancement(nn.Module):
    def __init__(self, hidden_size=768, audio_dim=5, vision_dim=20, num_heads=8, dropout=0.1):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.vision_proj = nn.Linear(vision_dim, hidden_size)
        self.audio_att = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.vision_att = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_features, audio, vision):
        a_ctx, _ = self.audio_att(text_features, self.audio_proj(audio), self.audio_proj(audio), need_weights=False)
        v_ctx, _ = self.vision_att(text_features, self.vision_proj(vision), self.vision_proj(vision), need_weights=False)
        enhanced = self.fusion(torch.cat([a_ctx, v_ctx], dim=-1))
        return self.norm(text_features + self.dropout(enhanced))

class CENET(nn.Module):
    def __init__(self, model_name="bert-base-uncased", injection_index=1, dropout=0.2):
        super().__init__()
        self.injection_index = injection_index
        self.bert = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden_size = self.bert.config.hidden_size
        self.ce = CrossModalEnhancement(hidden_size=hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, text_bert, audio, vision):
        target_layer = self.bert.encoder.layer[self.injection_index - 1]
        def inject_ce(module, inputs, output):
            hidden = output if isinstance(output, tuple) else output
            enhanced = self.ce(hidden, audio, vision)
            return (enhanced,) + output[1:] if isinstance(output, tuple) else enhanced

        handle = target_layer.register_forward_hook(inject_ce)
        try:
            outputs = self.bert(
                input_ids=text_bert[:, 0, :],
                attention_mask=text_bert[:, 1, :],
                token_type_ids=text_bert[:, 2, :],
                return_dict=True
            )
        finally:
            handle.remove()

        return self.regressor(self.dropout(outputs.last_hidden_state[:, 0, :])).squeeze(-1)

print("Baseline models defined.")
# 4. Early Fusion + Multi-Step CoT
class EarlyFusionCoT(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.text_pool = TemporalAttention(768, 64)
        self.audio_pool = TemporalAttention(5, 64)
        self.vision_pool = TemporalAttention(20, 64)
        self.text_enc = nn.Sequential(nn.Linear(768, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.audio_enc = nn.Sequential(nn.Linear(5, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.vision_enc = nn.Sequential(nn.Linear(20, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.mod_att = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.reasoning_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.step_norm = nn.LayerNorm(hidden_dim)
        self.step_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1)) for _ in range(3)])
        self.final_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 1)
        )

    def forward(self, text, audio, vision):
        tf = self.text_enc(self.text_pool(text))
        af = self.audio_enc(self.audio_pool(audio))
        vf = self.vision_enc(self.vision_pool(vision))
        mod_feats = torch.stack([tf, af, vf], dim=1)
        weights = torch.softmax(self.mod_att(mod_feats).squeeze(-1), dim=1)
        fused = self.fusion((mod_feats * weights.unsqueeze(-1)).reshape(tf.size(0), -1))

        hidden = fused
        step_preds = []
        for idx, feat in enumerate([tf, af, vf]):
            hidden = self.step_norm(hidden + self.reasoning_cell(feat, hidden))
            step_preds.append(self.step_heads[idx](hidden).squeeze(-1))

        final_pred = self.final_head(torch.cat([hidden, fused, tf], dim=-1)).squeeze(-1)
        return final_pred, step_preds, weights

# 5. Late Fusion + PEAR-CoT + CMFF
class PEARCoT_CMFF(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.text_enc = nn.Sequential(nn.Linear(768, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.20))
        self.audio_enc = nn.Sequential(nn.Linear(5, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, d), nn.LayerNorm(d), nn.GELU())
        self.vision_enc = nn.Sequential(nn.Linear(20, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, d), nn.LayerNorm(d), nn.GELU())
        self.text_att = TemporalAttention(d, d // 2)
        self.audio_att = TemporalAttention(d, d // 2)
        self.vision_att = TemporalAttention(d, d // 2)
        self.ta_att = nn.MultiheadAttention(d, 4, dropout=0.15, batch_first=True)
        self.tv_att = nn.MultiheadAttention(d, 4, dropout=0.15, batch_first=True)
        self.audio_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.vision_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.cmff = nn.Sequential(nn.Linear(d * 5, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.25), nn.Linear(256, d), nn.LayerNorm(d), nn.GELU())
        self.reason_query = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.reason_cell = nn.GRUCell(d, d)
        self.reason_norm = nn.LayerNorm(d)
        self.reason_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.regression = nn.Sequential(nn.Linear(d * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.25), nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 1))

    def forward(self, text, audio, vision):
        t, a, v = self.text_enc(text), self.audio_enc(audio), self.vision_enc(vision)
        tp, ap, vp = self.text_att(t), self.audio_att(a), self.vision_att(v)
        ta = self.text_att(self.ta_att(t, a, a, need_weights=False)[0])
        tv = self.text_att(self.tv_att(t, v, v, need_weights=False)[0])
        ga = self.audio_gate(torch.cat([tp, ta], dim=1))
        gv = self.vision_gate(torch.cat([tp, tv], dim=1))
        fused = self.cmff(torch.cat([tp, ga * ta + (1 - ga) * ap, gv * tv + (1 - gv) * vp, ta, tv], dim=1))

        state = fused
        for _ in range(3):
            state = self.reason_norm(self.reason_cell(self.reason_query(state), fused))
        gate = self.reason_gate(torch.cat([fused, state], dim=1))
        final_feat = gate * state + (1 - gate) * fused
        return self.regression(torch.cat([fused, final_feat], dim=1))

print("CoT neural models defined.")