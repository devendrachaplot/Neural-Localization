import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class Localization_2D_A3C(torch.nn.Module):

    def __init__(self, args):
        super(Localization_2D_A3C, self).__init__()

        self.map_size = args.map_size

        num_orientations = 4
        num_actions = 3
        n_policy_conv1_filters = 16
        n_policy_conv2_filters = 16
        size_policy_conv1_filters = 3
        size_policy_conv2_filters = 3
        self.action_emb_dim = 8
        self.depth_emb_dim = 8
        self.time_emb_dim = 8
        self.action_hist_size = args.hist_size

        conv_out_height = (((self.map_size - size_policy_conv1_filters) + 1) -
                           size_policy_conv2_filters) + 1
        conv_out_width = (((self.map_size - size_policy_conv1_filters) + 1) -
                          size_policy_conv2_filters) + 1

        self.policy_conv1 = nn.Conv2d(num_orientations + 1,
                                      n_policy_conv1_filters,
                                      size_policy_conv1_filters,
                                      stride=1)
        self.policy_conv2 = nn.Conv2d(n_policy_conv1_filters,
                                      n_policy_conv2_filters,
                                      size_policy_conv2_filters,
                                      stride=1)

        self.action_emb_layer = nn.Embedding(num_actions + 1,
                                             self.action_emb_dim)
        self.depth_emb_layer = nn.Embedding(args.map_size,
                                            self.depth_emb_dim)
        self.time_emb_layer = nn.Embedding(args.max_episode_length + 1,
                                           self.time_emb_dim)

        self.proj_layer = nn.Linear(
            n_policy_conv2_filters * conv_out_height * conv_out_width, 256)
        self.critic_linear = nn.Linear(
            256 + self.action_emb_dim * self.action_hist_size +
            self.depth_emb_dim + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(
            256 + self.action_emb_dim * self.action_hist_size +
            self.depth_emb_dim + self.time_emb_dim, num_actions)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (ax, dx, tx) = inputs
        conv_out = F.elu(self.policy_conv1(inputs))
        conv_out = F.elu(self.policy_conv2(conv_out))
        conv_out = conv_out.view(conv_out.size(0), -1)
        proj = self.proj_layer(conv_out)
        action_emb = self.action_emb_layer(ax)
        depth_emb = self.depth_emb_layer(dx)
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((
            proj,
            action_emb.view(-1, self.action_emb_dim * self.action_hist_size),
            depth_emb.view(-1, self.depth_emb_dim),
            time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x)
