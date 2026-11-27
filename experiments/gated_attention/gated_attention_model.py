"""Shared decoder only model

This backbone contains:

    token embedding E, 
    learned position embedding P (same logic as from the lecture), 
    attention with W_q / W_k / W_v,
    causal masking, 
    residual connections, 
    layer norm, 
    a feed-forward layer, 
    and the unembedding E^T (tied weights).  


Three attention variants share this single backbone:

    baseline : standard causal multi-head softmax attention
    g1       : headwise output gate.  g = sigmoid(W_g x) gives one scalar per head
               and token.
    sink     : one learned sink logit per head, added once as an extra softmax
               option for every query. The main idea is abstaining stright from the Richard Zhe Wang.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ('baseline', 'g1', 'sink')
MODEL_VERSION = 'tinygpt-v1'


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, variant='baseline', dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        assert variant in VARIANTS
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.variant = variant
        
        # queries, keys, values and output projection

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        if variant == 'g1':
            # one gate logit per head, from the hidden state at this token
            self.W_g = nn.Linear(d_model, n_head)
        if variant == 'sink':
            # one learned logit per head, value can be zero
            self.sink_logit = nn.Parameter(torch.zeros(n_head))
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, allowed, gate_override=None, return_attn=False):
        
        """
        x: [B, T, C] normalized hidden states; allowed: bool [B or 1, 1, T, T].
        """
        
        B, T, C = x.shape
        H, D = self.n_head, self.d_head
        Q = self.W_q(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        K = self.W_k(x).view(B, T, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, D).transpose(1, 2)

        similarities = Q @ K.transpose(-2, -1) / math.sqrt(D)  # [B, H, T, T]
        similarities = similarities.masked_fill(~allowed, float('-inf'))
        extra = {}
        if self.variant == 'sink':
            sink = self.sink_logit.view(1, H, 1, 1).expand(B, H, T, 1)
            sm = torch.softmax(torch.cat([similarities, sink], dim=-1), dim=-1)
            extra['sink_mass'] = sm[..., -1]           # [B, H, T]
            sm = sm[..., :-1]                          # phantom value = 0 -> drop it
        else:
            sm = torch.softmax(similarities, dim=-1)
        if return_attn:
            extra['attn'] = sm
        A = self.attn_dropout(sm) @ V                  # [B, H, T, D]

        gate = None
        if self.variant == 'g1':
            gate = torch.sigmoid(self.W_g(x))          # [B, T, H]
            if gate_override is not None:
                gate = torch.as_tensor(gate_override, dtype=x.dtype, device=x.device).expand(B, T, H)
            A = A * gate.transpose(1, 2).unsqueeze(-1)  # scale each head's output
        out = A.transpose(1, 2).reshape(B, T, C)       # concatenate heads
        return self.resid_dropout(self.W_o(out)), gate, extra


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, d_model, n_head, variant, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, variant, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x, allowed, gate_override=None, return_attn=False):
        a, gate, extra = self.attn(self.ln_1(x), allowed, gate_override, return_attn)
        x = x + a
        x = x + self.ffn(self.ln_2(x))
        return x, gate, extra


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, n_layer, n_head,
                 variant='baseline', dropout=0.0):
        super().__init__()
        self.variant = variant
        self.block_size = block_size
        self.n_layer, self.n_head = n_layer, n_head
        self.E = nn.Embedding(vocab_size, d_model)   # word embedding (unembedding = E^T)
        self.P = nn.Embedding(block_size, d_model)   # learned position embedding
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, variant, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():      # GPT-2 style scaled residual init
            if name.endswith('attn.W_o.weight') or name.endswith('ffn.2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, segment_start=None, gate_override=None,
                return_gates=False, return_attn=False):
        """idx: [B, T] token ids -> logits [B, T, vocab].

        segment_start: optional LongTensor [B].  Queries at positions >= s may
            only attend to keys at positions >= s (keys before s are masked in
            every layer).  Position ids are unchanged.
        gate_override: optional dict {layer: tensor broadcastable to [B, T, H]}.
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.E(idx) + self.P(pos))
        allowed = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril().view(1, 1, T, T)
        if segment_start is not None:
            q = pos.view(1, T, 1)
            k = pos.view(1, 1, T)
            s = segment_start.to(idx.device).view(B, 1, 1)
            allowed = allowed & ~((q >= s) & (k < s)).unsqueeze(1)   # [B, 1, T, T]
        gates, extras = [], []
        for layer, block in enumerate(self.blocks):
            override = None if gate_override is None else gate_override.get(layer)
            x, gate, extra = block(x, allowed, override, return_attn)
            gates.append(gate)
            extras.append(extra)
        logits = self.ln_f(x) @ self.E.weight.T
        if return_gates or return_attn:
            return logits, {'gates': gates, 'extras': extras}
        return logits


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    variant_only = sum(p.numel() for n, p in model.named_parameters()
                       if '.W_g.' in n or n.endswith('sink_logit'))
    return total, variant_only


def build_model(config, variant, init_seed):
    
    """Paired initialization: every shared parameter comes from a baseline model
    built with `init_seed`, so baseline / g1 / sink start from identical shared
    weights.  Variant-only parameters (W_g, sink_logit) come from a separate seed."""
    
    torch.manual_seed(init_seed)
    reference = TinyGPT(**config, variant='baseline')
    torch.manual_seed(init_seed + 10_000)
    model = TinyGPT(**config, variant=variant)
    missing, unexpected = model.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all('.W_g.' in n or n.endswith('sink_logit') for n in missing), missing
    return model


def save_checkpoint(path, model, config, variant, **extra):
    torch.save({'model_version': MODEL_VERSION, 'config': config, 'variant': variant,
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                **extra}, path)


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    assert ckpt['model_version'] == MODEL_VERSION, ckpt['model_version']
    model = TinyGPT(**ckpt['config'], variant=ckpt['variant'])
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return model.to(device).eval(), ckpt



def batch_windows(windows, idx, pad_id):
    """Pad a list of windows (rows of the saved windows table) into tensors."""
    rows = [windows[i] for i in idx]
    T = max(len(r['tokens']) - 1 for r in rows)
    x = np.full((len(rows), T), pad_id, dtype=np.int64)
    y = np.full((len(rows), T), -100, dtype=np.int64)    # -100 = not scored
    for b, r in enumerate(rows):
        tok = np.asarray(r['tokens'], dtype=np.int64)
        n = len(tok) - 1
        x[b, :n] = tok[:-1]
        y[b, r['score_from']:n] = tok[1 + r['score_from']:]
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def score_windows(model, windows, pad_id, device, batch_size=32,
                  gate_override_fn=None, keep_gates=False):
    """Per-target next-token loss (nats) for every scored target in `windows`.

    windows: list of dicts with tokens, score_from, story_id, offset.
    gate_override_fn(batch_index_list, T) -> dict for TinyGPT.forward, or None.
    Returns (rows, gates): rows has story_id, target_pos, loss; gates is a list
    of [L, T, H] float16 arrays (one per window) if keep_gates.
    """
    model.eval()
    story, tpos, losses, kept = [], [], [], []
    order = np.argsort([-len(w['tokens']) for w in windows])   # similar lengths together
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        x, y = batch_windows(windows, idx, pad_id)
        x, y = x.to(device), y.to(device)
        override = None if gate_override_fn is None else gate_override_fn(idx, x.shape[1])
        logits, info = model(x, gate_override=override, return_gates=True)
        loss = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction='none')
        loss, y = loss.cpu().numpy(), y.cpu().numpy()
        if keep_gates:
            g = torch.stack(info['gates'], dim=1).to(torch.float16).cpu().numpy()  # [B, L, T, H]
        for b, i in enumerate(idx):
            w = windows[i]
            pos = np.nonzero(y[b] != -100)[0]
            story.extend([w['story_id']] * len(pos))
            tpos.extend((w['offset'] + pos).tolist())   # story-token index of each target
            losses.extend(loss[b, pos].tolist())
            if keep_gates:
                kept.append((i, g[b, :, :len(w['tokens']) - 1]))
    rows = {'story_id': np.array(story), 'target_pos': np.array(tpos, dtype=np.int32),
            'loss': np.array(losses, dtype=np.float32)}
    gates = None
    if keep_gates:
        gates = [None] * len(windows)
        for i, g in kept:
            gates[i] = g
    return rows, gates
