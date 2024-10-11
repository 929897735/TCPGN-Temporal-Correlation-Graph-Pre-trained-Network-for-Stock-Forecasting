import torch
import torch.nn as nn
import numpy as np

from model.module import GAT, Transformer_Blocks, GraphDecoder


class TCGPN(nn.Module):
    def __init__(self, settings) -> None:
        """模型初始化.

        Args:
            settings: 配置文件.
        """
        super().__init__()

        self.settings = settings

        # embedding
        self.input_embedding_1 = nn.Linear(
            settings["input_dim"], settings["input_embedding_temporal_hidden_dim"]
        )

        self.input_embedding_2 = nn.Linear(
            settings["input_embedding_temporal_hidden_dim"],
            settings["input_embedding_temporal_out_dim"],
        )

        # GAT
        self.gat_1 = GAT(
            input_dim=settings["input_embedding_temporal_out_dim"],
            output_dim=settings["gat_output_dim"],
            num_head=settings["num_head"],
            time_step=settings["time_step"],
            drop_out=["gat_1_dropout"],
        )

        # Transformer
        self.transformer_encoder = Transformer_Blocks(
            input_dim=settings["gat_output_dim"],
            hidden_dim=settings["transformer_encoder_hidden_dim"],
            output_dim=settings["transformer_encoder_output_dim"],
            time_step=settings["time_step"],
            head_num=settings["transformer_encoder_head_num"],
            use_mask=settings["transformer_encoder_use_mask"],
        )

        self.transformer_encoder2 = Transformer_Blocks(
            input_dim=settings["transformer_encoder_output_dim"],
            hidden_dim=settings["transformer_encoder_hidden_dim"],
            output_dim=settings["transformer_encoder_output_dim"],
            time_step=settings["time_step"],
            head_num=settings["transformer_encoder_head_num"],
            use_mask=settings["transformer_encoder_use_mask"],
        )

        self.transformer_encoder3 = Transformer_Blocks(
            input_dim=settings["transformer_encoder_output_dim"],
            hidden_dim=settings["transformer_encoder_hidden_dim"],
            output_dim=settings["transformer_encoder_output_dim"],
            time_step=settings["time_step"],
            head_num=settings["transformer_encoder_head_num"],
            use_mask=settings["transformer_encoder_use_mask"],
        )

        # decoder
        self.transformer_decoder = Transformer_Blocks(
            input_dim=settings["transformer_encoder_output_dim"],
            hidden_dim=settings["transformer_decoder_hidden_dim"],
            output_dim=settings["transformer_decoder_output_dim"],
            time_step=settings["time_step"],
            head_num=settings["transformer_decoder_head_num"],
            use_mask=settings["transformer_decoder_use_mask"],
        )

        # graph decoder
        self.graph_decoder = GraphDecoder(
            input_dim=settings["transformer_encoder_output_dim"], time_step=settings["time_step"], hidden_dim=settings['gen_graph_hidden_dim'])

        # fc_layer
        self.fc_layer_out_1 = nn.Linear(
            settings["transformer_decoder_output_dim"], settings["output_hidden_dim"]
        )

        self.fc_layer_out_2 = nn.Linear(
            settings["output_hidden_dim"], settings["input_dim"]
        )

        self.gelu = nn.GELU()

    def position_encoding(self, inputs, time_step, input_dim, device="cuda"):
        position_encoding = np.array(
            [
                [
                    pos / np.power(10000, 2.0 * (j // 2) / input_dim)
                    for j in range(input_dim)
                ]
                for pos in range(time_step)
            ]
        )
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(
            position_encoding).to(inputs.device)
        inputs = inputs + position_encoding
        return inputs.to(torch.float32)

    def graph_generate(self, X):

        stock_num = X.shape[0]
        X_l = X.cpu().repeat(stock_num, 1).reshape(stock_num, stock_num, -1)
        X_r = X.cpu().repeat(1, stock_num).reshape(stock_num, stock_num, -1)
        a = (X_l - X_r).sum(dim=-1)
        return a.cuda()

    def forward(self, X, Graph, mode=1, X_tempral_label=None):
        X_temporal = self.input_embedding_1(X)

        X_temporal = self.position_encoding(
            X_temporal,
            self.settings["time_step"],
            self.settings["input_embedding_temporal_hidden_dim"],
        )

        X_temporal = self.input_embedding_2(X_temporal)

        X_temporal = self.gat_1(X_temporal, Graph)

        X_Encoder_out = self.transformer_encoder(X_temporal)
        X_Encoder_out = self.transformer_encoder2(X_Encoder_out)
        X_Encoder_out = self.transformer_encoder3(X_Encoder_out)

        X_temporal = self.transformer_decoder(X_Encoder_out)

        X_temporal = self.fc_layer_out_1(X_temporal)

        X_temporal = self.gelu(X_temporal)

        X_temporal = self.fc_layer_out_2(X_temporal)

        graph = self.graph_decoder(X_Encoder_out)

        return X_temporal, X_Encoder_out, graph


def temporal_loss(x_output, x_true, mask=None):
    if mask is None:
        return torch.square(x_output - x_true).reshape(-1).mean()
    else:
        x_shape = x_output.shape
        if mask.sum() == 0:
            return torch.square(x_output - x_true).reshape(-1).mean()
        else:
            return torch.square(x_output*mask - x_true*mask).reshape(-1).mean()*x_shape[0]/mask.reshape(-1).sum()


def graph_loss(graph_output, graph, mask=None):
    if mask is None:
        return torch.square(graph_output - graph/100).reshape(-1).mean()
    else:
        if mask.sum() == 0:
            return torch.square(graph_output - graph/100).reshape(-1).mean()
        else:
            return torch.square(graph_output*mask - graph/100*mask).reshape(-1).sum()/mask.reshape(-1).sum()


