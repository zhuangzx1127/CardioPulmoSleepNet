import torch
import numpy as np
from datetime import datetime, timedelta


class BaseExtractor:
    def __init__(self):
        pass

    def __call__(self, npz, start: int, end: int, value):
        raise NotImplementedError


class att2seq(BaseExtractor):
    def __init__(self, frame_per_token=1, dtype="float"):
        super().__init__()
        self.frame_per_token = frame_per_token
        self.dtype = dtype

    def __call__(self, npz, start: int, end: int, value):
        return (
            torch.ones(
                size=((end - start) * self.frame_per_token,),
                dtype=getattr(torch, self.dtype),
            )
            * value
        )

def default_extractor(name: str, frames_per_token: int, input_channels, dtype):
    if input_channels == 1:

        def extract(npz, start: int, end: int, value):
            if name in npz:
                result = torch.tensor(
                    npz[name].flatten()[start * frames_per_token : end * frames_per_token]
                ).to(dtype=getattr(torch, dtype))
                assert result.shape[0] == (end - start) * frames_per_token, f"{npz} 好像有点问题欸。"
                return result
            print(f"{npz}中其实不存在{name}！")
            return torch.zeros(
                size=((end - start) * frames_per_token,), dtype=getattr(torch, dtype)
            )

    else:

        def extract(npz, start: int, end: int, value):
            if name in npz:
                return torch.tensor(
                    npz[name][:, start * frames_per_token : end * frames_per_token].T  # seq_len * channel_num
                ).to(dtype=getattr(torch, dtype))
            return torch.zeros(
                ((end - start) * frames_per_token, input_channels),
                dtype=getattr(torch, dtype),
            )

    return extract

class stageRepeater(BaseExtractor):
    def __init__(self):
        self.extractor = default_extractor('stage5', 1, 1, 'long')
    def __call__(self, npz, start: int, end: int, value):
        return torch.repeat_interleave(self.extractor(npz, start, end, value), 30, 0)

class time_extractor(BaseExtractor):
    def __init__(self, interval=30, format="%H:%M:%S"):
        super().__init__()
        self.shift = interval / (24 * 3600) * 2 * torch.pi
        self.format = format

    def __call__(self, npz, start, end, value):
        t = datetime.strptime(value, self.format).time()
        seconds = t.hour * 3600 + t.minute * 60 + t.second
        start_angle = seconds / (24 * 3600) * 2 * torch.pi
        angles = torch.arange(start, end) * self.shift + start_angle
        return torch.stack([torch.sin(angles), torch.cos(angles)], dim=1)


class classification_time_extractor(BaseExtractor):
    def __init__(self, class_interval = 300, interval=30, format="%H:%M:%S"):
        super().__init__()
        self.shift = interval / class_interval
        self.format = format
        self.class_interval = class_interval

    def __call__(self, npz, start, end, value):
        t = datetime.strptime(value, self.format).time()
        seconds = t.hour * 3600 + t.minute * 60 + t.second
        start_angle = seconds / self.class_interval
        return torch.round((torch.arange(start, end) * self.shift + start_angle) % (24 * 3600 / self.class_interval))
     
        
class cls_classification_time_extractor(BaseExtractor):
    def __init__(self, interval=30, format="%H:%M:%S"):
        super().__init__()
        self.format = format
        self.interval = timedelta(seconds=interval)

    def __call__(self, npz, start, end, value):
        t = datetime.strptime(value, self.format) + (start + end) / 2 * self.interval
        return ((t.hour + 1) // 2) % 12
    
class cls_classification_time_extractor_higher_resolution(BaseExtractor):
    def __init__(self, class_interval = 300, interval=30, format="%H:%M:%S"):
        super().__init__()
        self.shift = interval / class_interval
        self.format = format
        self.class_interval = class_interval
        self.interval = timedelta(seconds=interval)

    def __call__(self, npz, start, end, value):
        t = datetime.strptime(value, self.format) + (start + end) / 2 * self.interval
        seconds = (t.hour + 1) % 24 * 3600 + t.minute * 60 + t.second  # 23点开始是0，省区后面的麻烦。
        return round(seconds / self.class_interval) % (24 * 3600 // self.class_interval) # 预测中心时刻
    
class cls_regression_time_extractor(BaseExtractor):
    def __init__(self, interval=30, format="%H:%M:%S"):
        super().__init__()
        self.format = format
        self.interval = timedelta(seconds=interval)

    def __call__(self, npz, start, end, value):
        t = datetime.strptime(value, self.format) + (start + end) / 2 * self.interval
        hour = (t.hour + t.minute / 60 + t.second / 3600 + 1) % 24 / 2 # 23点开始是0，省区后面的麻烦。
        return hour