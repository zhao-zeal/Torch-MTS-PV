"""From
https://github.com/zezhishao/BasicTS/blob/master/baselines/DLinear/arch/dlinear.py
https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
"""

import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x):
        # padding on the both ends of time series
        # 把时间序列两端用首尾值做复制填充，然后在时间维上做平均池化，得到一个平滑后的序列
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block"""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean # 季节/残差项 趋势项

# individual = False 所有变量共享一套趋势映射和季节映射参数
# individual = True 模型为每个变量各建一套线性层
class DLinear(nn.Module):
    """
    Paper: Are Transformers Effective for Time Series Forecasting?
    Link: https://arxiv.org/abs/2205.13504
    Official Code: https://github.com/cure-lab/DLinear
    """

    def __init__(self, 
                 enc_in=321,
                 seq_len=336, 
                 pred_len=96, 
                 individual=False, 
                 kernel_size=49,
                 ):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, history_data) -> torch.Tensor:
        """Feed forward of DLinear.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        assert history_data.shape[-1] == 1  # only use the target feature
        x = history_data[..., 0]  # B, L, N
        seasonal_init, trend_init = self.decompsition(x) # 分解
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1) # [B, N, L]
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init) 

        prediction = seasonal_output + trend_output
        return prediction.permute(0, 2, 1).unsqueeze(-1)  # [B, L, N, 1]

if __name__ == "__main__":
    from torchinfo import summary
    model = DLinear()
    summary(model, [64, 336, 321, 1])
    
