import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, num_head, time_step, drop_out, alpha=0.01) -> None:
        super().__init__()
        self.drop_out = drop_out
        self.alpha = alpha
        self.num_head = num_head

        self.timestep_aggregation = nn.Linear(time_step, 1)

        self.time_step = time_step

        self.fc = nn.Linear(input_dim, num_head*output_dim)
        self.a = nn.Parameter(torch.Tensor(num_head, output_dim*2))

        nn.init.xavier_uniform_(self.fc.weight.data, gain=1.414)  
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x: torch.tensor, adj: torch.tensor):
        numnodes = x.shape[0]

        x = self.fc(x)
        x = x.view(numnodes, self.time_step, self.num_head, -1)
        edges = adj.nonzero(as_tuple=False)

        x_flat = self.timestep_aggregation(x.permute(0, 2, 3, 1)).squeeze()
        x_flat = x_flat.view(numnodes, self.num_head, -1)
        edge_indices_row = edges[:, 0]
        edge_indices_col = edges[:, 1]

        a_input = torch.cat(
            [
                torch.index_select(
                    input=x_flat, index=edge_indices_row, dim=0),
                torch.index_select(
                    input=x_flat, index=edge_indices_col, dim=0),
            ],
            dim=-1,
        )

        attn_logits = torch.einsum("bhc,hc->bh", a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        attn_matrix = attn_logits.new_zeros(
            adj.shape + (self.num_head,)).fill_(-9e15)
        attn_matrix[adj[..., None].repeat(
            1, 1, self.num_head) != 0] = attn_logits.reshape(-1)

        attn_probs = F.softmax(attn_matrix, dim=-2)
        x = torch.einsum("ijh,jthc->ithc", attn_probs, x)
        x = x.mean(dim=-2).squeeze()
        return x


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)

        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Transformer_Blocks(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, time_step, head_num, use_mask=False
    ) -> None:
        super().__init__()

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.output_dim = output_dim

        self.head_num = head_num

        self.use_mask = use_mask


        self.residual_layer = nn.Linear(
            in_features=self.input_dim, out_features=self.output_dim
        )

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=-1)


        self.score_mask = torch.ones((time_step, time_step))

        self.score_mask = nn.Parameter(
            torch.tril(self.score_mask, diagonal=0)
        ).requires_grad_(False)

        self.padding_mask = nn.Parameter(
            self.score_mask.eq(0).to(torch.float32) * 1e9
        ).requires_grad_(False)


        self.q_layer = nn.ModuleList()
        self.k_layer = nn.ModuleList()
        self.v_layer = nn.ModuleList()

        for _ in range(self.head_num):

            self.q_layer.append(
                nn.Linear(in_features=self.input_dim,
                          out_features=self.hidden_dim)
            )

            self.k_layer.append(
                nn.Linear(in_features=self.input_dim,
                          out_features=self.hidden_dim)
            )

            self.v_layer.append(
                nn.Linear(in_features=self.input_dim,
                          out_features=self.hidden_dim)
            )

        self.outputs_layer = nn.Linear(
            in_features=self.hidden_dim * self.head_num, out_features=self.output_dim
        )

        self.outputs_norm = nn.LayerNorm(self.output_dim)

        self.outputs_fc_layer = nn.Linear(
            in_features=self.output_dim, out_features=self.output_dim
        )

        self.outputs_fc_norm = nn.LayerNorm(self.output_dim)

    def forward(self, inputs):
        residual = self.residual_layer(inputs)

        q, k, v = {}, {}, {}
        att_scores, att_weights, head_outputs = {}, {}, {}
        for h in range(self.head_num):
            q[h] = self.relu(self.q_layer[h](inputs))
            k[h] = self.relu(self.k_layer[h](inputs))
            v[h] = self.tanh(self.v_layer[h](inputs))
            att_scores[h] = torch.matmul(q[h], k[h].transpose(-1, -2))
            att_scores[h] /= torch.sqrt(
                torch.tensor(self.hidden_dim, dtype=torch.float32)
            )

            if self.use_mask:

                att_scores[h] *= self.score_mask

                att_scores[h] -= self.padding_mask

            att_weights[h] = self.softmax(att_scores[h])

            head_outputs[h] = torch.matmul(att_weights[h], v[h])

        if self.head_num > 1:
            outputs = torch.concat([head_outputs[h]
                                   for h in head_outputs], dim=-1)
            outputs = self.tanh(self.outputs_layer(outputs))
        else:
            outputs = head_outputs[0]

        outputs = self.outputs_norm(outputs + residual)

        outputs_fc = self.relu(self.outputs_fc_layer(outputs))

        outputs = self.outputs_fc_norm(outputs_fc + outputs)

        return outputs


class GraphDecoder(nn.Module):
    def __init__(self, time_step, input_dim, hidden_dim) -> None:
        super().__init__()
        self.time_step_mix_1 = nn.Linear(time_step, 1)
        self.time_step_mix_2 = nn.Linear(time_step, 1)
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        left = self.time_step_mix_1(x.permute(0, 2, 1)).squeeze()
        right = self.time_step_mix_2(x.permute(0, 2, 1)).squeeze()

        left = self.fc_1(left)
        right = self.fc_2(right)
        adj = torch.matmul(left, right.permute(1, 0))

        return adj
