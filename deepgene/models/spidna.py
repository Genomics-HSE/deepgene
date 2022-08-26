#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class SpidnaLightning(LightningModule):
    
    def __init__(self):
        super(SpidnaLightning, self).__init__()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.kl_div(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer


class SPIDNABlock(nn.Module):
    def __init__(self, num_output, num_feature):
        super(SPIDNABlock, self).__init__()
        self.num_output = num_output
        self.phi = nn.Conv2d(num_feature * 2, num_feature, (1, 3))
        self.phi_bn = nn.BatchNorm2d(num_feature * 2)
        self.maxpool = nn.MaxPool2d((1, 2))
        self.fc = nn.Linear(num_output, num_output)
    
    def forward(self, x, output):
        x = self.phi(self.phi_bn(x))
        psi1 = torch.mean(x, 2, keepdim=True)
        psi = psi1
        current_output = self.fc(torch.mean(psi[:, :self.num_output, :, :], 3).squeeze(2))
        output = output + current_output
        psi = psi.expand(-1, -1, x.size(2), -1)
        x = torch.cat((x, psi), 1)
        x = F.relu(self.maxpool(x))
        
        return x, output


class SPIDNA(SpidnaLightning):
    def __init__(self, num_output, num_block, num_feature, device, weights_path=None, **kwargs):
        super(SPIDNA, self).__init__()
        self.save_hyperparameters()
        self.num_output = num_output
        self.conv_pos = nn.Conv2d(1, num_feature, (1, 3))
        self.conv_pos_bn = nn.BatchNorm2d(num_feature)
        self.conv_snp = nn.Conv2d(1, num_feature, (1, 3))
        self.conv_snp_bn = nn.BatchNorm2d(num_feature)
        self.blocks = nn.ModuleList([SPIDNABlock(num_output, num_feature) for i in range(num_block)])
        
        # if weights_path is not None:
        #     if device == 'cpu':
        #         state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['state_dict']
        #     else:
        #         state_dict = torch.load(weights_path)['state_dict']
        #     self.load_state_dict(state_dict)
    
    def forward(self, x):
        pos = x[:, 0, :].view(x.shape[0], 1, 1, -1)
        snp = x[:, 1:, :].unsqueeze(1)
        pos = F.relu(self.conv_pos_bn(self.conv_pos(pos))).expand(-1, -1, snp.size(2), -1)
        snp = F.relu(self.conv_snp_bn(self.conv_snp(snp)))
        x = torch.cat((pos, snp), 1)
        output = torch.zeros(x.size(0), self.num_output)
        for block in self.blocks:
            x, output = block(x, output)
        
        return output


class SPIDNA2Block_adaptive(LightningModule):
    def __init__(self, num_output, num_feature):
        super(SPIDNA2Block_adaptive, self).__init__()
        self.num_output = num_output
        self.phi = nn.Conv2d(num_feature * 2, num_feature, (1, 3))
        self.phi_gn = nn.GroupNorm(1, num_feature * 2)
        self.maxpool = nn.MaxPool2d((1, 2))
        self.fc = nn.Linear(num_output, num_output)
    
    def forward(self, x, output):
        x = self.phi_gn(x)
        x = self.phi(x)
        psi = torch.mean(x, 2, keepdim=True)
        current_output = self.fc(torch.mean(psi[:, :self.num_output, :, :], 3).squeeze(2))
        output = output + current_output
        psi = psi.expand(-1, -1, x.size(2), -1)
        x = torch.cat((x, psi), 1)
        x = F.relu(self.maxpool(x))
        
        return x, output


class SPIDNA2_adaptive(LightningModule):
    def __init__(self, num_output, num_block, num_feature, device, **kwargs):
        super(SPIDNA2_adaptive, self).__init__()
        self.num_output = num_output
        self.conv_pos = nn.Conv2d(1, num_feature, (1, 3))
        self.conv_pos_in = nn.InstanceNorm2d(num_feature)
        self.conv_snp = nn.Conv2d(1, num_feature, (1, 3))
        self.conv_snp_in = nn.InstanceNorm2d(num_feature)
        self.blocks = nn.ModuleList([SPIDNA2Block_adaptive(num_output, num_feature) for i in range(num_block)])
        self.device = device
    
    def forward(self, x):
        pos = x[:, 0, :].view(x.shape[0], 1, 1, -1)
        snp = x[:, 1:, :].unsqueeze(1)
        pos = F.relu(self.conv_pos_in(self.conv_pos(pos))).expand(-1, -1, snp.size(2), -1)
        snp = F.relu(self.conv_snp_in(self.conv_snp(snp)))
        x = torch.cat((pos, snp), 1)
        output = torch.zeros(x.size(0), self.num_output).to(self.device)
        for block in self.blocks:
            x, output = block(x, output)
        
        return output
