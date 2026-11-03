from torch import nn
import torch


class seqProj(nn.Module):
    def __init__(
        self,
        output_name,
        hidden_size,
        num_classes: int = 5,
        token_len: int = 1,
        softmax: bool = False,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.num_classes = num_classes
        self.fc = nn.Linear(hidden_size, self.num_classes * self.token_len)
        self.softmax = softmax
        self.output_name = output_name

    def forward(self, x, **kwargs):
        x = x[:, 1:, :]
        x = self.fc(x)
        if self.token_len != 1:
            x = x.reshape(x.shape[0], -1, self.num_classes)
        if self.num_classes == 1 and self.softmax:
            return {self.output_name: torch.sigmoid(x).squeeze(-1)}
        elif self.num_classes == 1:
            return {self.output_name: x.squeeze(-1)}
        elif self.softmax:
            return {self.output_name: torch.softmax(x, dim=-1)}
        else:
            return {self.output_name: x}


class clsProj(nn.Module):
    def __init__(
        self,
        output_name,
        hidden_size,
        num_classes: int = 5,
        token_len: int = 1,
        softmax: bool = False,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = softmax
        self.output_name = output_name
        self.num_classes = num_classes

    def forward(self, x, **kwargs):
        x = self.fc(x[:, 0, :])
        if self.num_classes == 1 and self.softmax:
            return {self.output_name: torch.sigmoid(x).squeeze(-1)}
        elif self.num_classes == 1:
            return {self.output_name: x.squeeze(-1)}
        elif self.softmax:
            return {self.output_name: torch.softmax(x, dim=-1)}
        else:
            return {self.output_name: x}


class seqMLP(nn.Module):
    def __init__(
        self,
        output_name,
        hidden_size,
        num_classes: int = 5,
        token_len: int = 1,
        softmax: bool = False,
        hidden_layers=None,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.num_classes = num_classes
        self.output_name = output_name
        self.softmax = softmax

        # 动态创建隐藏层
        layers = []
        input_dim = hidden_size
        if hidden_layers is not None:
            for h in hidden_layers:
                layers.append(nn.Linear(input_dim, h))
                layers.append(nn.ReLU())
                input_dim = h

        # 添加最后输出层
        layers.append(nn.Linear(input_dim, self.num_classes * self.token_len))
        self.fc = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = x[:, 1:, :]
        x = self.fc(x)
        if self.token_len != 1:
            x = x.reshape(x.shape[0], -1, self.num_classes)
        if self.num_classes == 1 and self.softmax:
            return {self.output_name: torch.sigmoid(x).squeeze(-1)}
        elif self.num_classes == 1:
            return {self.output_name: x.squeeze(-1)}
        elif self.softmax:
            return {self.output_name: torch.softmax(x, dim=-1)}
        else:
            return {self.output_name: x}


class clsMLP(nn.Module):
    def __init__(
        self,
        output_name,
        hidden_size,
        num_classes: int = 5,
        softmax: bool = False,
        hidden_layers=[],
    ) -> None:
        super().__init__()
        self.output_name = output_name
        self.softmax = softmax
        self.num_classes = num_classes

        # 动态创建隐藏层
        layers = []
        input_dim = hidden_size
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        # 添加最后输出层
        layers.append(nn.Linear(input_dim, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.fc(x[:, 0, :])
        if self.num_classes == 1 and self.softmax:
            return {self.output_name: torch.sigmoid(x).squeeze(-1)}
        elif self.num_classes == 1:
            return {self.output_name: x.squeeze(-1)}
        elif self.softmax:
            return {self.output_name: torch.softmax(x, dim=-1)}
        else:
            return {self.output_name: x}


class bmiProj(nn.Module):
    def __init__(
        self,
        output_name,
        hidden_size,
        num_classes: int = 5,
        token_len: int = 1,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.num_classes = num_classes
        self.fc = nn.Linear(hidden_size, self.num_classes * self.token_len)
        self.softmax = softmax
        self.output_name = output_name

    def forward(self, x, **kwargs):
        x = x[:, 1:, :]
        x = self.fc(x)

        return {self.output_name: torch.sigmoid(x).squeeze(-1) * 80 + 10}

class timeProj(nn.Module):
    def __init__(
        self,
        output_name,
        hidden_size,
        num_classes: int = 5,
        token_len: int = 1,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        # self.token_len = token_len
        # self.num_classes = num_classes
        self.fc = nn.Linear(hidden_size, 1)
        self.softmax = softmax
        self.output_name = output_name

    def forward(self, x, **kwargs):
        x = x[:, 0, :]
        x = self.fc(x)

        return {self.output_name: torch.sigmoid(x).squeeze(-1) * 12}