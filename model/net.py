# import math
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from model.utils import log_normal_density

# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.shape[0], 1, -1)


# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
#         self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
#         self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         batch_size, channels, length = x.size()
#         q = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)
#         k = self.key(x).view(batch_size, -1, length)
#         v = self.value(x).view(batch_size, -1, length)

#         attention = torch.bmm(q, k)
#         attention = F.softmax(attention, dim=-1)
#         out = torch.bmm(v, attention.permute(0, 2, 1))
#         out = out.view(batch_size, channels, length)
#         out = self.gamma * out + x
#         return out


# class CNNPolicy(nn.Module):
#     def __init__(self, frames, action_space):
#         super(CNNPolicy, self).__init__()
#         self.logstd = nn.Parameter(torch.zeros(action_space))

#         self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
#         self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
#         self.act_attention = SelfAttention(32)
#         self.act_fc1 = nn.Linear(128 * 32, 256)
#         self.act_fc2 = nn.Linear(256 + 2 + 2, 128)
#         self.actor1 = nn.Linear(128, 1)
#         self.actor2 = nn.Linear(128, 1)

#         self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
#         self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
#         self.crt_attention = SelfAttention(32)
#         self.crt_fc1 = nn.Linear(128 * 32, 256)
#         self.crt_fc2 = nn.Linear(256 + 2 + 2, 128)
#         self.critic = nn.Linear(128, 1)

#     def forward(self, x, goal, speed):
#         """
#             returns value estimation, action, log_action_prob
#         """
#         # action
#         a = F.relu(self.act_fea_cv1(x))
#         a = F.relu(self.act_fea_cv2(a))
#         a = self.act_attention(a)
#         a = a.view(a.shape[0], -1)
#         a = F.relu(self.act_fc1(a))

#         a = torch.cat((a, goal, speed), dim=-1)
#         a = F.relu(self.act_fc2(a))
#         mean1 = F.sigmoid(self.actor1(a))
#         mean2 = F.tanh(self.actor2(a))
#         mean = torch.cat((mean1, mean2), dim=-1)

#         logstd = self.logstd.expand_as(mean)
#         std = torch.exp(logstd)
#         action = torch.normal(mean, std)

#         # action prob on log scale
#         logprob = log_normal_density(action, mean, std=std, log_std=logstd)

#         # value
#         v = F.relu(self.crt_fea_cv1(x))
#         v = F.relu(self.crt_fea_cv2(v))
#         v = self.crt_attention(v)
#         v = v.view(v.shape[0], -1)
#         v = F.relu(self.crt_fc1(v))
#         v = torch.cat((v, goal, speed), dim=-1)
#         v = F.relu(self.crt_fc2(v))
#         v = self.critic(v)

#         return v, action, logprob, mean

#     def evaluate_actions(self, x, goal, speed, action):
#         v, _, _, mean = self.forward(x, goal, speed)
#         logstd = self.logstd.expand_as(mean)
#         std = torch.exp(logstd)
#         # evaluate
#         logprob = log_normal_density(action, mean, log_std=logstd, std=std)
#         dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
#         dist_entropy = dist_entropy.sum(-1).mean()
#         return v, logprob, dist_entropy


# class MLPPolicy(nn.Module):
#     def __init__(self, obs_space, action_space):
#         super(MLPPolicy, self).__init__()
#         # action network
#         self.act_fc1 = nn.Linear(obs_space, 64)
#         self.act_fc2 = nn.Linear(64, 128)
#         self.mu = nn.Linear(128, action_space)
#         self.mu.weight.data.mul_(0.1)
#         # torch.log(std)
#         self.logstd = nn.Parameter(torch.zeros(action_space))

#         # value network
#         self.value_fc1 = nn.Linear(obs_space, 64)
#         self.value_fc2 = nn.Linear(64, 128)
#         self.value_fc3 = nn.Linear(128, 1)
#         self.value_fc3.weight.data.mul(0.1)

#     def forward(self, x):
#         """
#             returns value estimation, action, log_action_prob
#         """
#         # action
#         act = self.act_fc1(x)
#         act = F.tanh(act)
#         act = self.act_fc2(act)
#         act = F.tanh(act)
#         mean = self.mu(act)  # N, num_actions
#         logstd = self.logstd.expand_as(mean)
#         std = torch.exp(logstd)
#         action = torch.normal(mean, std)

#         # value
#         v = self.value_fc1(x)
#         v = F.tanh(v)
#         v = self.value_fc2(v)
#         v = F.tanh(v)
#         v = self.value_fc3(v)

#         # action prob on log scale
#         logprob = log_normal_density(action, mean, std=std, log_std=logstd)
#         return v, action, logprob, mean

#     def evaluate_actions(self, x, action):
#         v, _, _, mean = self.forward(x)
#         logstd = self.logstd.expand_as(mean)
#         std = torch.exp(logstd)
#         # evaluate
#         logprob = log_normal_density(action, mean, log_std=logstd, std=std)
#         dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
#         dist_entropy = dist_entropy.sum(-1).mean()
#         return v, logprob, dist_entropy


# if __name__ == '__main__':
#     net = CNNPolicy(3, 2)
#     observation = Variable(torch.randn(2, 3))
#     v, action, logprob, mean = net.forward(observation)
#     print(v)

    

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.utils import log_normal_density

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], 1, -1)
    

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, length = x.size()
        q = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, length)
        v = self.value(x).view(batch_size, -1, length)

        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, length)
        out = self.gamma * out + x
        return out

class ResidualConvBlock(nn.Module):
    """轻量级残差卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn(self.conv1(x)))
        x = self.bn(self.conv2(x))
        x += residual
        return F.relu(x)

class CNNPolicy(nn.Module):
    def __init__(self, frames, action_space):
        super(CNNPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        # 修改后的残差卷积结构
        self.act_conv = nn.Sequential(
            ResidualConvBlock(frames, 32, kernel_size=5, stride=2),
            ResidualConvBlock(32, 32, kernel_size=3, stride=2),
            SelfAttention(32),  # 保留注意力但减少计算量
            nn.AdaptiveAvgPool1d(4)  # 新增全局池化减少参数
        )
        
        # 调整全连接层尺寸
        self.act_fc1 = nn.Linear(32*4, 128)  # 原为128*32=4096
        self.act_fc2 = nn.Linear(128+2+2, 64)  # 缩小中间层
        
        # 修改动作头结构
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_space)
        )

        # 价值网络采用相同结构
        self.crt_conv = nn.Sequential(
            ResidualConvBlock(frames, 32, kernel_size=5, stride=2),
            ResidualConvBlock(32, 32, kernel_size=3, stride=2),
            SelfAttention(32),
            nn.AdaptiveAvgPool1d(4)
        )
        
        self.crt_fc1 = nn.Linear(32*4, 128)
        self.crt_fc2 = nn.Linear(128+2+2, 64)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # 参数初始化调整
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)  # 增大初始化方差
                nn.init.constant_(m.bias, 0.1)  # 避免dead neuron
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, goal, speed):
        # 动作路径
        a = self.act_conv(x)
        a = a.view(a.size(0), -1)
        a = F.relu(self.act_fc1(a))
        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean = torch.sigmoid(self.actor(a))  # 统一输出激活
        
        # 价值路径
        v = self.crt_conv(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.crt_fc1(v))
        v = torch.cat((v, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        # 动作分布处理
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action):
        v, _, _, mean = self.forward(x, goal, speed)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


class MLPPolicy(nn.Module):
    def __init__(self, obs_space, action_space):
        super(MLPPolicy, self).__init__()
        # action network
        self.act_fc1 = nn.Linear(obs_space, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.mu = nn.Linear(128, action_space)
        self.mu.weight.data.mul_(0.1)
        # torch.log(std)
        self.logstd = nn.Parameter(torch.zeros(action_space))

        # value network
        self.value_fc1 = nn.Linear(obs_space, 64)
        self.value_fc2 = nn.Linear(64, 128)
        self.value_fc3 = nn.Linear(128, 1)
        self.value_fc3.weight.data.mul(0.1)

    def forward(self, x):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        act = self.act_fc1(x)
        act = F.tanh(act)
        act = self.act_fc2(act)
        act = F.tanh(act)
        mean = self.mu(act)  # N, num_actions
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # value
        v = self.value_fc1(x)
        v = F.tanh(v)
        v = self.value_fc2(v)
        v = F.tanh(v)
        v = self.value_fc3(v)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, action):
        v, _, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


if __name__ == '__main__':
    net = CNNPolicy(3, 2)
    observation = Variable(torch.randn(2, 3))
    v, action, logprob, mean = net.forward(observation)
    print(v)

    

