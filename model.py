import torch
import torch.nn as nn
import timm
class PetNetONLY(nn.Module):
    def __init__(self, model_name, out_features, inp_channels, pretrained):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, 256)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, image):
        embeddings = self.model(image)
        x = self.dropout(embeddings)
        output = self.fc(x)
        return output
        