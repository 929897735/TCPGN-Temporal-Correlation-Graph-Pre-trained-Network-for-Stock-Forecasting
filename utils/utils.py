import torch
from torch.utils.data import dataset


class DataGenerator(dataset.Dataset):
    def __init__(
        self,
        date_begin,
        date_end,
        features,
        labels,
        time_step,
        reg_capital=None,
        stock_sample_time=10,
        stock_sample_num=4000,
        mask_sample_time=50,
        temporal_mask_ratio=0.3,
        graph_mask_ratio=0.5,
        graph_type='industry',
        turnover_pos=None,
        industry_pos=None,
        vali_label_pos=None,
        use_rand=True,
    ) -> None:
        super().__init__()
        self.len_date_list = date_end - date_begin

        self.date_begin = date_begin
        self.date_end = date_end

        self.features = features[:, date_begin - (time_step - 1): date_end, :]

        self.labels = labels[:, date_begin - (time_step - 1):date_end, :]

        self.time_step = time_step

        self.stock_num, self.totol_time_step, self.feature_num = self.features.shape

        self.reg_capital = reg_capital

        self.stock_sample_time = stock_sample_time
        self.stock_sample_num = stock_sample_num
        self.mask_sample_time = mask_sample_time

        self.graph_type = graph_type
        if self.graph_type == 'industry':
            self._cal_reg_capital_graph()

        self.mask_len = int(self.time_step*temporal_mask_ratio)

        self.temporal_mask_ratio = temporal_mask_ratio
        self.graph_mask_ratio = graph_mask_ratio

        self.turnover_pos = turnover_pos
        self.industry_pos = industry_pos
        self.vali_label_pos = vali_label_pos

        self.stock_sample = torch.randperm(self.stock_num)[
            :self.stock_sample_num]

        self.use_rand = use_rand

        self.normalize()

    def normalize(self):

        self.norm_mean = torch.mean(self.features, dim=0).unsqueeze(dim=0)
        self.norm_std = torch.std(self.features, dim=0).unsqueeze(dim=0)
        self.norm_features = (
            self.features - self.norm_mean) / (self.norm_std + 1e-12)
        self.norm_features = self.norm_features.type(torch.float32)

    def _cal_reg_capital_graph(self):
        ci = torch.tensor(self.reg_capital).reshape(-1, 1)
        cj = ci.permute(1, 0)
        ci = 1 / (ci + 1)
        self.cij = ci * cj

    def _cal_industry_graph(self, features, labels, stock_sample):

        turnover_feature = features[:, -1, self.turnover_pos].reshape(-1, 1)
        turnover_feature_T = turnover_feature.permute(1, 0)
        turnover_feature = 1 / turnover_feature

        turnover_feature = torch.where(
            torch.isinf(turnover_feature),
            torch.full_like(turnover_feature, 0),
            turnover_feature,
        )
        turnover_tensor = torch.nan_to_num(
            turnover_feature * turnover_feature_T, 0)


        industry_feature = features[:, -1, self.industry_pos].reshape(-1, 1)
        industry_feature = industry_feature.repeat(
            industry_feature.shape[1], industry_feature.shape[0]
        )
        industry_same_tensor = torch.eq(
            industry_feature, industry_feature.permute(1, 0)
        ).to(torch.int)

        industry_graph = self.cij[stock_sample, stock_sample] + turnover_tensor


        industry_graph = industry_graph*industry_same_tensor

        validity_label = labels[:, -1, self.vali_label_pos].reshape(-1, 1)
        validity_graph = validity_label * validity_label.permute(1, 0)

        industry_graph = industry_graph*validity_graph

        return industry_graph
    
    def _cal_euclidean_graph(self, features, labels):
        graph=torch.sqrt( torch.square(features.unsqueeze(dim=1)-features.unsqueeze(dim=0)).mean(dim=-1).sum(dim=-1))

        validity_label = labels[:, -1, self.vali_label_pos].reshape(-1, 1)
        validity_graph = validity_label * validity_label.permute(1, 0)

        graph = graph*validity_graph

        return graph 

    def _normalize_adjacency_matrix(self, adjacency_matrix):
        out_degree = torch.sum(adjacency_matrix, dim=1)
        out_degree_inv = torch.pow(out_degree, -1)
        out_degree_inv[out_degree_inv == float('inf')] = 0.
        D_out_inv = torch.diag(out_degree_inv)
        normalized_adjacency_matrix = torch.matmul(D_out_inv, adjacency_matrix)

        normalized_adjacency_matrix = (normalized_adjacency_matrix-torch.mean(
            normalized_adjacency_matrix, dim=(0, 1)))/torch.std(normalized_adjacency_matrix, dim=(0, 1))

        return normalized_adjacency_matrix

    def __getitem__(self, index):
        t = index//(self.mask_sample_time *
                    self.stock_sample_time)+self.time_step
        if index % self.mask_sample_time == 0:
            if self.use_rand:
                self.stock_sample = torch.randperm(self.stock_num)[
                    :self.stock_sample_num]
            else:
                self.stock_sample = torch.arange(self.stock_num)[
                    :self.stock_sample_num]
        batch_features = self.norm_features[self.stock_sample,
                                            t - self.time_step: t, :]
        batch_labels = self.labels[self.stock_sample, t-1: t, :]

        tem_mask_tensor = 0
        batch_mask_features = 0
        if self.mask_len != 0:
            rand_pos = torch.randint(
                0, self.time_step-self.mask_len, (self.stock_sample_num, 1))
            idx_pos = torch.arange(self.stock_sample_num).unsqueeze(1)
            tem_mask_tensor = torch.ones_like(batch_features)
            tem_mask_tensor[idx_pos, rand_pos+torch.arange(self.mask_len)] = 0
            batch_mask_features = batch_features*tem_mask_tensor

        graph = 0
        mask_graph = 0
        graph_mask_tensor = 0
        if self.graph_type == 'industry':
            graph = self._cal_industry_graph(
                batch_features, batch_labels, self.stock_sample)
            graph = self._normalize_adjacency_matrix(graph)
            if self.graph_mask_ratio != 0:
                mask_pos = torch.randperm(self.stock_sample_num*self.stock_sample_num)[
                    :int(self.graph_mask_ratio*self.stock_sample_num*self.stock_sample_num)]
                graph_mask_tensor = torch.ones_like(graph).reshape(-1)
                graph_mask_tensor[mask_pos] = 0
                graph_mask_tensor = graph_mask_tensor.reshape(
                    self.stock_sample_num, self.stock_sample_num)
                mask_graph = graph*graph_mask_tensor
        if self.graph_type == 'l2dis':
            graph = self._cal_euclidean_graph(
                batch_features, batch_labels)
            graph = self._normalize_adjacency_matrix(graph)
            if self.graph_mask_ratio != 0:
                mask_pos = torch.randperm(self.stock_sample_num*self.stock_sample_num)[
                    :int(self.graph_mask_ratio*self.stock_sample_num*self.stock_sample_num)]
                graph_mask_tensor = torch.ones_like(graph).reshape(-1)
                graph_mask_tensor[mask_pos] = 0
                graph_mask_tensor = graph_mask_tensor.reshape(
                    self.stock_sample_num, self.stock_sample_num)
                mask_graph = graph*graph_mask_tensor

        return batch_features, batch_labels, batch_mask_features, tem_mask_tensor, graph, mask_graph, graph_mask_tensor

    def __len__(self):
        return self.len_date_list*self.mask_sample_time*self.stock_sample_time
