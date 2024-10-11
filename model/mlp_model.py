import torch
import torch.nn as nn

class mlp_model(nn.Module):
    def __init__(self, settings, pretrain_model) -> None:
        super().__init__()
        self.pretrain_model = pretrain_model

        self.pretrain_model.requires_grad_ = False

        self.residual_layer = nn.Linear(
            in_features=settings["transformer_encoder_output_dim"], out_features=settings["fc_layer_2_output_dim"])


        self.fc_score_1 = nn.Linear(
            in_features=settings["fc_layer_2_output_dim"], out_features=settings["score_hidden_dim"]
        )
        self.fc_score_2 = nn.Linear(
            in_features=settings["score_hidden_dim"], out_features=1
        )
        self.soft_max = nn.Softmax(dim=-2)

        self.fc_layer_1 = nn.Linear(
            settings["transformer_encoder_output_dim"],
            settings["fc_layer_1_output_dim"],
        )
        self.relu = nn.ReLU()
        self.fc_layer_2 = nn.Linear(
            settings["fc_layer_1_output_dim"], settings["fc_layer_2_output_dim"]
        )

        self.output_layer = nn.Linear(
            in_features=settings["fc_layer_2_output_dim"], out_features=settings["fc_layer_2_output_dim"]
        )
        self.output_layer_2 = nn.Linear(
            in_features=settings["fc_layer_2_output_dim"], out_features=1
        )
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.fc_layer_1.weight)
        nn.init.xavier_normal_(self.fc_layer_2.weight)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.xavier_normal_(self.output_layer_2.weight)
        nn.init.xavier_normal_(self.fc_score_1.weight)
        nn.init.xavier_normal_(self.fc_score_2.weight)
        nn.init.xavier_normal_(self.residual_layer.weight)

    def forward(self, x, graph):
        _, X_Encoder_out, _ = self.pretrain_model(
            x, graph, mode=1)
        
        out = X_Encoder_out

        residual = self.residual_layer(out)

        out = self.tanh(self.fc_layer_1(out))

        out = self.tanh(self.fc_layer_2(out))

        out = out+residual

        out_score = self.fc_score_1(out)

        out_score = self.fc_score_2(out_score)

        out_score = self.soft_max(out_score)

        out = torch.sum(out_score * out, dim=-2)

        output=self.output_layer_2(self.relu(self.output_layer))

        return output
