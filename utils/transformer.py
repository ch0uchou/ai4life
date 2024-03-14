import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
    output, attention_weights
    """

    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = torch.tensor(k).shape[-1].float()
    dk_sqrt = torch.sqrt(dk)
    scaled_attention_logits = matmul_qk / dk_sqrt

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)

    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, d_ff, activation):
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        getattr(F, activation)(),  # (batch_size, seq_len, dff)
        nn.Linear(d_ff, d_model)  # (batch_size, seq_len, d_model)
    )

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, depth, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = depth

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Assuming scaled_dot_product_attention is defined and converted to PyTorch
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)

        return output, attention_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, activation, **kwargs):
        super(TransformerEncoderLayer, self).__init__(*kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // self.num_heads

        self.mha = MultiHeadAttention(self.d_model, self.num_heads, self.depth)
            
        self.ffn = point_wise_feed_forward_network(self.d_model, self.d_ff, self.activation)

        self.layernorm1 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x, None)  # (batch_size, input_seq_len, d_model)
            
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, activation, n_layers):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads,
                                                                     d_ff, dropout, activation) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.encoder_layers[i](x)

        return x
    
class PatchClassEmbedding(nn.Module):
    def __init__(self, d_model, n_patches, pos_emb=None, kernel_initializer=None):
        super(PatchClassEmbedding, self).__init__()
        self.d_model = d_model
        self.n_tot_patches = n_patches + 1
        self.pos_emb = pos_emb

        self.class_embed = nn.Parameter(torch.randn(1, 1, self.d_model))
        if self.pos_emb is not None:
            self.pos_emb = torch.tensor(np.load(self.pos_emb))
            self.lap_position_embedding = nn.Embedding(self.pos_emb.shape[0], self.d_model)
        else:
            self.position_embedding = nn.Embedding(self.n_tot_patches, self.d_model)

    def forward(self, inputs):
        x =  self.class_embed.repeat(inputs.shape[0], 1, 1)
        x = torch.cat((x, inputs), dim=1)
        if self.pos_emb is None:
            positions = torch.arange(0, self.n_tot_patches)
            pe = self.position_embedding(positions)
        else:
            pe = self.pos_emb
            pe = pe.view(1, -1)
            pe = self.lap_position_embedding(pe)
        encoded = x + pe
        return encoded