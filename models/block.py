import torch
import torch.nn as nn
import torch.nn.functional as F

# class Conv2D(nn.Module):
#     def __init__(self, in_channels, out_channels, equal=False):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = nn.LeakyReLU(0.1)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act(x)
#         return x
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x
        return out
    
class CoordGate(nn.Module):
    def __init__(self, channels, coord_channels=2):
        super(CoordGate, self).__init__()
        self.coord_fc = nn.Sequential(
            nn.Linear(coord_channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid()  # Ensure gating values are between 0 and 1
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Create coordinate grid
#         coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)),  dim=-1)
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij'), dim=-1)
        coords = coords.float().to(x.device).view(-1, 2)  # Flatten
        coords = coords.unsqueeze(0).expand(b, -1, -1)  # Expand batch size
        
        # Apply fully connected layers
        gating_map = self.coord_fc(coords).view(b, h, w, c).permute(0, 3, 1, 2)
        
        # Apply gating map
        x = x * gating_map
        
        return x
    
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, equal=False, padding=1):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0)
        self.pad = nn.ZeroPad2d(padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.coord_gate = CoordGate(out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.coord_gate(x)
        x = self.act(x)
        return x

class ConvTrans2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
        self.coord_gate = CoordGate(out_channels)
    
    def forward(self, x):
        x = self.convT(x)
        x = self.bn(x)
        x = self.coord_gate(x)
        x = self.act(x)
        return x
    
# class ResNet(nn.Module):
#     def __init__(self, channels):
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm2d(channels)
#         self.coord_gate = CoordGate(channels)
        
#     def forward(self, x):
#         residual = x
#         x = self.conv1(x)
#         x = self.bn(x)
#         x = self.coord_gate(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = x + residual
#         return x
class ResNet(nn.Module):
    def __init__(self, channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x + residual
        return x


class GatedFeatureFusionModule(nn.Module):
    def __init__(self, in_channels, side_channels):
        super(GatedFeatureFusionModule, self).__init__()
        self.bn_main = nn.BatchNorm2d(in_channels)
        self.bn_aux = nn.BatchNorm2d(side_channels)
        
        self.Wg = nn.Conv2d(in_channels + side_channels, in_channels, kernel_size=1)
        self.Wf = nn.Conv2d(in_channels + side_channels, in_channels, kernel_size=1)
        self.Wo = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.coord_gate = CoordGate(in_channels)
        
    def forward(self, main_feature, aux_feature):
        main_feature = self.bn_main(main_feature)
        aux_feature = self.bn_aux(aux_feature)
        
        combined = torch.cat([main_feature, aux_feature], dim=1)
        
        gate = torch.sigmoid(self.coord_gate(self.Wg(combined)))
        refinement = self.coord_gate(self.Wf(combined))
        
        output = self.coord_gate(self.Wo(gate * main_feature + (1 - gate) * refinement))
        return output
    

# class GatedFeatureFusionModule(nn.Module):
#     def __init__(self, in_channels, side_channels):
#         super(GatedFeatureFusionModule, self).__init__()
#         self.bn_main = nn.BatchNorm2d(in_channels)
#         self.bn_aux = nn.BatchNorm2d(side_channels)
        
#         self.Wg = nn.Conv2d(in_channels + side_channels, in_channels, kernel_size=1)
#         self.Wf = nn.Conv2d(in_channels + side_channels, in_channels, kernel_size=1)
#         self.Wo = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
#     def forward(self, main_feature, aux_feature):
#         main_feature = self.bn_main(main_feature)
#         aux_feature = self.bn_aux(aux_feature)
        
#         combined = torch.cat([main_feature, aux_feature], dim=1)
        
#         gate = torch.sigmoid(self.Wg(combined))
#         refinement = self.Wf(combined)
        
#         output = self.Wo(gate * main_feature + (1 - gate) * refinement)
#         return output