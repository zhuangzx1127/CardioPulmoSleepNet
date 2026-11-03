import lightning as L
import torch
import torch.nn as nn
from transformers import RoFormerConfig, RoFormerModel
import typing as t
from wuji_dl.factory import get_class
from wuji_dl.ops.functional import length_mask
from models.base import base_model
from wuji_dl.ops.res_block1d import ResBlock1d

from sympy.ntheory import factorint
#from peft import LoraConfig, get_peft_model
import pdb

import re
class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        channels: int,
        layers: list[int],
        token_len: int,
        norm_layer: t.Callable[[int], nn.Module] = nn.BatchNorm1d,
        embedding_size: int = 512,
    ) -> None:
        super().__init__()
        self._norm_layer = norm_layer
        self.inplanes = embedding_size // 8
        strides = sorted(
            sum([[k] * v for k, v in factorint(token_len).items()], [])
        )
        while len(strides) > 5:
            strides.append(strides[0] * strides[1])
            strides.pop(0)
            strides.pop(0)
            strides = sorted(strides)
        for _ in range(5 - len(strides)):
            strides.append(1)
        self.layer0 = nn.Sequential(
            nn.Conv1d(
                channels, self.inplanes, 7, stride=strides[0], padding=3, bias=False
            ),
            norm_layer(self.inplanes),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=strides[1], padding=1),
        )
        self.layer1 = self._make_layer(embedding_size // 8, layers[0])
        self.layer2 = self._make_layer(embedding_size // 4, layers[1], strides[2])
        self.layer3 = self._make_layer(embedding_size // 2, layers[2], strides[3])
        self.layer4 = self._make_layer(embedding_size, layers[3], strides[4])
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, 1, stride=stride, bias=False),
                self._norm_layer(planes),
            )
        layers = [ResBlock1d(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResBlock1d(planes, planes, 1, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #pdb.set_trace()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SleepNetZeroHead(nn.Module):
    def __init__(
        self, head, token_len, input_channels, embedding_size, **kwargs
    ) -> None:
        super().__init__()
        self.mode = head

        self.feature_extractor = ResNetFeatureExtractor(
            input_channels,
            [2, 2, 2, 2],
            token_len,
            embedding_size=embedding_size,
        )

        self.input_channels = input_channels

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        if self.input_channels == 1:
           
            x = kwargs[self.mode][:, None].float()
        else:
            x = kwargs[self.mode].transpose(-1, -2).float()
        return self.feature_extractor(x).transpose(1, 2)


class SleepNetZeroBackbone(nn.Module):
    def __init__(
        self, features: list[dict] = [], embedding_size: int = 512, **kwargs
    ) -> None:
        super().__init__()
        self.feature_extractors = nn.ModuleList(
            SleepNetZeroHead(
                **args, embedding_size=embedding_size
            )
            for args in features
        )
        print(f"模型接受的输入：{','.join([f['head'] for f in features])}")
        config = RoFormerConfig(vocab_size=1, embedding_size=embedding_size, **kwargs)
        self.roformer = RoFormerModel(config)
        self.cls = nn.Parameter(torch.empty(1, 1, embedding_size))
        nn.init.normal_(self.cls)
        
        self.hidden_size = config.hidden_size

    def _forward(self, length: torch.Tensor, **kwargs):
        #pdb.set_trace()
        feats = [head(**kwargs) for head in self.feature_extractors]
        x = torch.stack(feats).sum(dim=0)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        mask = length_mask(x.size(1), length + 1)
        #pdb.set_trace()
        return self.roformer(
            inputs_embeds=x, attention_mask=mask, output_hidden_states=True
        )

    def get_hidden_states(self, **kwargs):
        return self._forward(**kwargs).hidden_states

    def forward(self, **kwargs):
        
        return self._forward(**kwargs).last_hidden_state

class MatrixNetwork(nn.Module):
    def __init__(self, size=120):
        super(MatrixNetwork, self).__init__()
        self.size = size
        # 定义一个可学习的权重矩阵，初始化为单位矩阵
        self.weight = nn.Parameter(torch.eye(size))


    def forward(self, x):
        # 将输入与矩阵相乘
        return x @ self.weight

class SleepNetZeroPretrained(nn.Module):
    def __init__(
        self, pretrain_task: str, pretrained: str, unfreeze_backbone: bool | str = True
    ):
        super().__init__()
        task_cls = get_class(L.LightningModule, pretrain_task)
        ckpt = task_cls.load_from_checkpoint(pretrained, map_location="cpu")

        
        self.pretrained: SleepNetZeroBackbone = ckpt.model.backbone
        #pdb.set_trace()
        if isinstance(ckpt.model, base_model):
            self.fc = ckpt.model.heads[0]  # 注释掉了 .fc 现在返回一个字典
            self.RNFExtractor_snore = ResNetFeatureExtractor(1, [2,2,2,2], 400)
            self.RNFExtractor_heartbeat = ResNetFeatureExtractor(1, [2,2,2,2], 50)
            self.RNFExtractor_breath = ResNetFeatureExtractor(1, [2,2,2,2], 2)
        else:
            self.fc = ckpt.model.fc
            self.RNFExtractor_snore = ResNetFeatureExtractor(1, [2,2,2,2], 12000)
            self.RNFExtractor_heartbeat = ResNetFeatureExtractor(1, [2,2,2,2], 1500)
            self.RNFExtractor_breath = ResNetFeatureExtractor(1, [2,2,2,2], 60)
        self.cls = ckpt.model.backbone.cls
        self.feature_extractors = ckpt.model.backbone.feature_extractors
        self.roformer = ckpt.model.backbone.roformer
        self.gate_snore = nn.Parameter(torch.ones((1,512)))
        self.gate_breath = nn.Parameter(torch.ones((1,512)))
        self.gate_heartbeat = nn.Parameter(torch.ones((1,512)))




        assert isinstance(self.pretrained, SleepNetZeroBackbone)
        config = self.pretrained.roformer.config

        self.hidden_size = config.hidden_size

        
        if type(unfreeze_backbone) == bool:
            if not unfreeze_backbone:
                for param in self.pretrained.parameters():
                    param.requires_grad = False
        else:
            pattern = re.compile(unfreeze_backbone)
            for name, param in self.pretrained.named_parameters():
                if not re.search(pattern, name):
                    param.requires_grad = False


    def forward(self, **kwargs):
        #pdb.set_trace()
        snore = kwargs['snore_bandpass'].unsqueeze(1) # 
        breath = kwargs['breath_bandpass'].unsqueeze(1) # 
        heartbeat = kwargs['heartbeat_bandpass'].unsqueeze(1) #
        length = kwargs['length']
        kwargs = {'breath': kwargs['breath'], 'heartbeat': kwargs['heartbeat']}    

        
        
        snore_checked = self.RNFExtractor_snore.forward(snore).squeeze(1).permute(0,2,1) # 
        breath_checked = self.RNFExtractor_breath.forward(breath).squeeze(1).permute(0,2,1) # 
        heartbeat_checked = self.RNFExtractor_heartbeat.forward(heartbeat).squeeze(1).permute(0,2,1) #

        #pdb.set_trace()
        hr_br = [head(**kwargs) for head in self.feature_extractors]
        x = torch.stack(hr_br).sum(dim=0)
        x = x + snore_checked * self.gate_snore  + heartbeat_checked * self.gate_heartbeat + breath_checked * self.gate_breath #
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        mask = length_mask(x.size(1), length + 1)
        grand_feats = self.roformer(inputs_embeds=x, attention_mask=mask, output_hidden_states=True)
        return self.fc(grand_feats.last_hidden_state)
        # grand_feats = self.pretrained.forward(**kwargs)
        # return self.fc(grand_feats)
        
    


class SleepNetZeroFacade:
    def __new__(cls, **kwargs):
        if "pretrained" in kwargs:
           
            return SleepNetZeroPretrained(**kwargs)
        return SleepNetZeroBackbone(**kwargs)

