import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from logger import Loggers

Logger = Loggers.get_logger(__name__)


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class VesuviusModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg
        self.th = None
        self.encoder = getattr(smp, cfg["Proto"])(**cfg['args'])

    def forward(self, image):
        x = self.encoder(image)
        # x = x.squeeze(-1)
        return x


class VesuviusSynthModel:
    def __init__(self, use_tta=False):
        self.models = []
        self.use_tta = use_tta

    def __call__(self, x):
        outputs = [torch.sigmoid(model(x)).to('cpu').numpy()
                   for model in self.models]
        avg_preds = np.mean(outputs, axis=0)
        return avg_preds

    def add_model(self, model):
        self.models.append(model)

# def build_model(cfg, weight="imagenet"):
#     Logger.info(f"model_name: {cfg['model_name']}")
#     Logger.info(f"backbone: {cfg['backbone']}")
#     if cfg['resume']:
#         model_path = get_saved_model_path(cfg)
#         if os.path.exists(model_path):
#             Logger.info(f'load model from: {model_path}')
#             _model = CustomModel(cfg, weight=None)
#             loaded_model = torch.load(model_path)
#             # print(loaded_model)
#             _model.load_state_dict(loaded_model['model'])
#             # best_loss = loaded_model['best_loss']
#             # best_loss = None if loaded_model['best_loss'] is None else loaded_model['best_loss']
#             best_loss = loaded_model['best_loss']
#             th = loaded_model['th'] if 'th' in loaded_model else 0.5
#             _model.th = th
#             return _model, best_loss
#         Logger.info(f'trained model not found')
#
#     if is_kaggle:
#         weight = None
#     _model = CustomModel(cfg, weight)
#     return _model, None
