# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from torch import nn
from torch.nn import functional as F


class LoRA(nn.Module):
    def __init__(
        self,
        dim,
        r,
        scaling=1.,
        dropout=0.,
        targets='qkvo',
    ):
        super().__init__()
        self.targets = targets
        self.dim = dim
        self.scaling = scaling
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
        if 'q' in targets:
            self.a_q = nn.Linear(dim, r, bias=False)
            self.b_q = nn.Linear(r, dim, bias=False)
        if 'k' in targets:
            self.a_k = nn.Linear(dim, r, bias=False)
            self.b_k = nn.Linear(r, dim, bias=False)
        if 'v' in targets:
            self.a_v = nn.Linear(dim, r, bias=False)
            self.b_v = nn.Linear(r, dim, bias=False)
        if 'o' in targets:
            self.a_o = nn.Linear(dim, r, bias=False)
            self.b_o = nn.Linear(r, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for n, m in self.named_modules():
            if 'a' in n:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if 'b' in n:
                nn.init.zeros_(m.weight)

    def forward_qkv(self, attn, x, identity):
        qkv = F.linear(x, attn.attn.in_proj_weight, attn.attn.in_proj_bias)
        if 'q' in self.targets:
            q_r = self.b_q(self.a_q(self.dropout(x))) * self.scaling
            qkv[:, :, : self.dim] += q_r
        if 'k' in self.targets:
            k_r = self.b_k(self.a_k(self.dropout(x))) * self.scaling
            qkv[:, :, self.dim : -self.dim] += k_r
        if 'v' in self.targets:
            v_r = self.b_v(self.a_v(self.dropout(x))) * self.scaling
            qkv[:, :, -self.dim :] += v_r
        N, L, C = qkv.shape
        qkv = qkv.view(N, L, 3, C//3).permute(2, 0, 1, 3).reshape(3*N, L, C//3)
        out = F.linear(qkv, attn.attn.out_proj.weight, attn.attn.out_proj.bias)
        if 'o' in self.targets:
            o_r = self.b_o(self.a_o(self.dropout(qkv))) * self.scaling
            out += o_r
        q, k, v = out.tensor_split(3, dim=0)
        v += identity
        return q, k, v

    def forward(self, attn, x, identity):
        assert attn.batch_first
        assert not attn.attn.batch_first
        assert attn.attn.dropout == 0
        x = x.transpose(0, 1)
        tgt_len, bsz, embed_dim = x.shape
        assert attn.embed_dims == embed_dim
        assert embed_dim == self.dim
        q, k, v = F.linear(x, attn.attn.in_proj_weight, attn.attn.in_proj_bias).chunk(3, dim=-1)
        if 'q' in self.targets:
            q = q.contiguous()
            q_r = self.b_q(self.a_q(self.dropout(x))) * self.scaling
            q += q_r
        if 'k' in self.targets:
            k = k.contiguous()
            k_r = self.b_k(self.a_k(self.dropout(x))) * self.scaling
            k += k_r
        if 'v' in self.targets:
            v = v.contiguous()
            v_r = self.b_v(self.a_v(self.dropout(x))) * self.scaling
            v += v_r
        q = q.contiguous().view(tgt_len, bsz * attn.attn.num_heads, attn.attn.head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * attn.attn.num_heads, attn.attn.head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * attn.attn.num_heads, attn.attn.head_dim).transpose(0, 1)

        attn_output, _ = F._scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        out = F.linear(attn_output, attn.attn.out_proj.weight, attn.attn.out_proj.bias)
        if 'o' in self.targets:
            o_r = self.b_o(self.a_o(self.dropout(attn_output))) * self.scaling
            out += o_r
        out = out.view(tgt_len, bsz, out.size(1))
        out = out.transpose(0, 1)

        return identity + attn.dropout_layer(attn.proj_drop(out))
