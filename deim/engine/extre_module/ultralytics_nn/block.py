# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch     
import torch.nn as nn   
import torch.nn.functional as F   
from functools import partial     
from timm.layers import DropPath, to_2tuple

from ..torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad 
from ..custom_nn.mlp.SEFN import SEFN

try:
    from flash_attn.flash_attn_interface import flash_attn_func     
    FLASH_ATTN_FLAG = True 
except ImportError as e:   
    # assert False, "import FlashAttention error! Please install FlashAttention first."  
    FLASH_ATTN_FLAG = False

__all__ = (
    "C1",
    "C2",  
    "C3", 
    "C2f",
    "C2fAttn",   
    "ImagePoolingAttn", 
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",   
    "Proto",   
    "RepC3", 
    "ResNetLayer",
    "ELAN1",
    "AConv",   
    "SPPELAN",  
    "C3k",
    "C3k2",
    "C2fPSA",     
    "C2PSA",
    "RepVGGDW", 
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",   
    "SCDown",
    "ABlock",
    "A2C2f"
)     

class C1(nn.Module):     
    """CSP Bottleneck with 1 convolution."""
    
    def __init__(self, c1, c2, n=1):     
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__() 
        self.cv1 = Conv(c1, c2, 1, 1)     
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))
   
    def forward(self, x):  
        """Applies cross-convolutions to input in the C3 module."""     
        y = self.cv1(x)
        return self.m(y) + y

    
class C2(nn.Module):  
    """CSP Bottleneck with 2 convolutions."""     

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()    
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)   
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))   

    def forward(self, x):   
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))     

     
class C2f(nn.Module): 
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""     
 
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""   
        super().__init__()  
        self.c = int(c2 * e)  # hidden channels     
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)   
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
  
    def forward(self, x):  
        """Forward pass through C2f layer."""     
        y = list(self.cv1(x).chunk(2, 1))    
        y.extend(m(y[-1]) for m in self.m) 
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

   
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""  

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values.""" 
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)   
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2) 
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))  
     
    def forward(self, x): 
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))   

     
class C3x(C3):   
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)  
        self.c_ = int(c2 * e)     
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))  
     
     
class RepC3(nn.Module):
    """Rep C3."""
 
    def __init__(self, c1, c2, n=3, e=1.0):     
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels     
        self.cv1 = Conv(c1, c2, 1, 1)   
        self.cv2 = Conv(c1, c2, 1, 1)     
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()  

    def forward(self, x):     
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))   

    
class C3Ghost(C3): 
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5): 
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))     
   

class GhostBottleneck(nn.Module):    
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__() 
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw   
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw 
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )    
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )
 
    def forward(self, x):  
        """Applies skip connection and concatenation to input tensor."""   
        return self.conv(x) + self.shortcut(x)     

    
class Bottleneck(nn.Module):  
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels     
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2     

    def forward(self, x):  
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

     
class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""    

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):   
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""    
        super().__init__() 
        c_ = int(c2 * e)  # hidden channels   
        self.cv1 = Conv(c1, c_, 1, 1)     
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)   
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)  
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)    
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):   
        """Applies a CSP bottleneck with 3 convolutions."""     
        y1 = self.cv3(self.m(self.cv1(x)))     
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))  
  

class ResNetBlock(nn.Module):    
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)   
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)    
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()     

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x)) 


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""   
    
    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first
  
        if self.is_first:
            self.layer = nn.Sequential(    
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     
            )
        else:   
            blocks = [ResNetBlock(c1, c2, s, e=e)]   
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks) 

    def forward(self, x): 
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):   
    """Max Sigmoid attention block.""" 
    
    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None     
        self.gl = nn.Linear(gc, ec)  
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0     
   
    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)   
        guide = guide.view(bs, -1, self.nh, self.hc)    
        embed = self.ec(x) if self.ec is not None else x   
        embed = embed.view(bs, self.nh, self.hc, h, w) 

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]  
        aw = aw / (self.hc**0.5)   
        aw = aw + self.bias[None, :, None, None]   
        aw = aw.sigmoid() * self.scale
   
        x = self.proj_conv(x)  
        x = x.view(bs, self.nh, -1, h, w) 
        x = x * aw.unsqueeze(2)    
        return x.view(bs, -1, h, w)    
  
    
class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""    
 
    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):  
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""  
        super().__init__() 
        self.c = int(c2 * e)  # hidden channels  
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)   
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)    
   
    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)     
        y.append(self.attn(y[-1], guide))  
        return self.cv2(torch.cat(y, 1))
   
    def forward_split(self, x, guide):     
        """Forward pass using split() instead of chunk().""" 
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))   
        return self.cv2(torch.cat(y, 1))

    
class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()  
     
        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))   
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))     
        self.proj = nn.Linear(ec, ct)    
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0    
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch]) 
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh 
        self.nf = nf    
        self.hc = ec // nh    
        self.k = k   

    def forward(self, x, text):     
        """Executes attention mechanism on input tensor x and guide tensor."""     
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]    
        x = torch.cat(x, dim=-1).transpose(1, 2) 
        q = self.query(text)   
        k = self.key(x) 
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)  
        q = q.reshape(bs, -1, self.nh, self.hc)  
        k = k.reshape(bs, -1, self.nh, self.hc)  
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)
     
        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)  
        x = self.proj(x.reshape(bs, -1, self.ec)) 
        return x * self.scale + text


class ContrastiveHead(nn.Module):     
    """Implements contrastive learning head for region-text similarity in vision-language models."""
   
    def __init__(self):     
        """Initializes ContrastiveHead with specified region-text similarity parameters."""    
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses     
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())
 
    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias

   
class BNContrastiveHead(nn.Module): 
    """   
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:     
        embed_dims (int): Embed dimensions of text and image features. 
    """
  
    def __init__(self, embed_dims: int): 
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses    
        self.bias = nn.Parameter(torch.tensor([-10.0])) 
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))   

    def forward(self, x, w):
        """Forward function of contrastive learning."""     
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)    
        x = torch.einsum("bchw,bkc->bkhw", x, w) 
        return x * self.logit_scale.exp() + self.bias 
   
     
class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels   
        self.cv1 = RepConv(c1, c_, k[0], 1) 


class RepCSP(C3):     
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""  
     
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):   
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):   
    """CSP-ELAN."""   
    
    def __init__(self, c1, c2, c3, c4, n=1): 
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""    
        super().__init__()  
        self.c = c3 // 2    
        self.cv1 = Conv(c1, c3, 1, 1)    
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1)) 
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)   

    def forward(self, x):    
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))  
   
    def forward_split(self, x): 
        """Forward pass using split() instead of chunk().""" 
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4): 
    """ELAN1 module with 4 convolutions."""
     
    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes.""" 
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)   
        self.cv2 = Conv(c3 // 2, c4, 3, 1) 
        self.cv3 = Conv(c4, c4, 3, 1)     
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
     
    
class AConv(nn.Module):
    """AConv."""   
  
    def __init__(self, c1, c2):  
        """Initializes AConv module with convolution layers."""   
        super().__init__()     
        self.cv1 = Conv(c1, c2, 3, 2, 1)     
 
    def forward(self, x):     
        """Forward pass through AConv layer.""" 
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)    


class ADown(nn.Module): 
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)  
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)    
   
    def forward(self, x): 
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1) 
        x1 = self.cv1(x1) 
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)    
        x2 = self.cv2(x2) 
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):     
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__() 
        self.c = c3 
        self.cv1 = Conv(c1, c3, 1, 1)   
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)    
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)   
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])     
        return self.cv5(torch.cat(y, 1))  

  
class CBLinear(nn.Module): 
    """CBLinear."""
   
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)
   
    def forward(self, x):    
        """Forward pass through CBLinear layer."""    
        return self.conv(x).split(self.c2s, dim=1)  


class CBFuse(nn.Module):
    """CBFuse."""    

    def __init__(self, idx):  
        """Initializes CBFuse module with layer index for selective feature fusion."""  
        super().__init__()
        self.idx = idx
    
    def forward(self, xs):     
        """Forward pass through CBFuse layer."""    
        target_size = xs[-1].shape[2:] 
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])] 
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)   

     
class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):   
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion. 
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels   
        self.cv1 = Conv(c1, c_, 1, 1)     
        self.cv2 = Conv(c1, c_, 1, 1)  
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)    
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]  
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))

   
class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
  
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):   
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""    
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )    
 

class C3k(C3):   
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""    
   
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):    
        """Initializes the C3k module with specified channels, number of layers, and configurations."""     
        super().__init__(c1, c2, n, shortcut, g, e)     
        c_ = int(c2 * e)  # hidden channels   
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
     
     
class RepVGGDW(torch.nn.Module):  
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""
    
    def __init__(self, ed) -> None:  
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()    
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False) 
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)  
        self.dim = ed
        self.act = nn.SiLU()    

    def forward(self, x):
        """   
        Performs a forward pass of the RepVGGDW block.   

        Args:
            x (torch.Tensor): Input tensor.
   
        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.   
        """   
        return self.act(self.conv(x) + self.conv1(x))
     
    def forward_fuse(self, x): 
        """    
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.
   
        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """   
        return self.act(self.conv(x))
     
    @torch.no_grad()
    def fuse(self):  
        """     
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly. 
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)  

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight     
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])
 
        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b  

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)  

        self.conv = conv 
        del self.conv1   
    
    
class CIB(nn.Module):  
    """  
    Conditional Identity Block (CIB) module.
  
    Args:    
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False. 
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):    
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),     
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),  
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),    
        )

        self.add = shortcut and c1 == c2 
  
    def forward(self, x):
        """  
        Forward pass of the CIB module. 
    
        Args:    
            x (torch.Tensor): Input tensor.

        Returns:     
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)
 

class C2fCIB(C2f):   
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.    
  
    Args:     
        c1 (int): Number of input channels.  
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1. 
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.    
        lk (bool, optional): Whether to use local key connection. Defaults to False.    
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.     
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.     
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""   
        super().__init__(c1, c2, n, shortcut, g, e)     
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
    
 
class Attention(nn.Module): 
    """    
    Attention module that performs self-attention on the input tensor.
   
    Args:
        dim (int): The input tensor dimension.     
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.
     
    Attributes:
        num_heads (int): The number of attention heads.   
        head_dim (int): The dimension of each attention head.     
        key_dim (int): The dimension of the attention key.   
        scale (float): The scaling factor for the attention scores.  
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.    
    """ 

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):   
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()  
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  
        self.key_dim = int(self.head_dim * attn_ratio)   
        self.scale = self.key_dim**-0.5    
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
    
    def forward(self, x): 
        """  
        Forward pass of the Attention module.    

        Args:
            x (torch.Tensor): The input tensor.     
  
        Returns:
            (torch.Tensor): The output tensor after self-attention.    
        """  
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(    
            [self.key_dim, self.key_dim, self.head_dim], dim=2 
        )     

        attn = (q.transpose(-2, -1) @ k) * self.scale  
        attn = attn.softmax(dim=-1)   
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
   

class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.
  
    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.     

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.  
  
    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers. 

    Examples:  
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)     
        >>> input_tensor = torch.randn(1, 128, 32, 32)   
        >>> output_tensor = psablock(input_tensor)   
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:    
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__() 

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut   
   
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor.""" 
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x   
  

class PSA(nn.Module):
    """     
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to   
    input tensors, enhancing feature extraction and processing capabilities.
   
    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.    
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.   

    Methods:  
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.   

    Examples: 
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)    
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """
   
    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()  
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)  

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))   

    def forward(self, x):     
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""  
        a, b = self.cv1(x).split((self.c, self.c), dim=1) 
        b = b + self.attn(b)
        b = b + self.ffn(b)   
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing. 

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.
    
    Attributes:   
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.     
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.  
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.
   
    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
     
    Examples:    
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)  
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__() 
        assert c1 == c2     
        self.c = int(c1 * e)   
        self.cv1 = Conv(c1, 2 * self.c, 1, 1) 
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor.""" 
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)     
        return self.cv2(torch.cat((a, b), 1)) 
    
  
class C2fPSA(C2f):  
    """ 
    C2fPSA module with enhanced feature extraction using PSA blocks.
 
    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:    
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.    
    
    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:     
        >>> import torch     
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):   
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""     
        assert c1 == c2  
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))

    
class SCDown(nn.Module):  
    """
    SCDown module for downsampling with separable convolutions.
  
    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.  

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels. 
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods: 
        forward: Applies the SCDown module to the input tensor.    

    Examples:     
        >>> import torch 
        >>> from ultralytics import SCDown  
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128) 
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """
  
    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()  
        self.cv1 = Conv(c1, c2, 1, 1)   
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""   
        return self.cv2(self.cv1(x))

class AAttn(nn.Module): 
    """  
    Area-attention module with the requirement of flash attention.

    Attributes:   
        dim (int): Number of hidden channels;  
        num_heads (int): Number of heads into which the attention mechanism is divided;   
        area (int, optional): Number of areas the feature map is divided. Defaults to 1. 
  
    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.    

    Examples:     
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)   
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64. 
 
    """ 

    def __init__(self, dim, num_heads, area=1):   
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()     
        self.area = area  
     
        self.num_heads = num_heads 
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads   

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)   
     
        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)   
  
  
    def forward(self, x):  
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W
     
        if x.is_cuda and FLASH_ATTN_FLAG:
            qk = self.qk(x).flatten(2).transpose(1, 2)
            v = self.v(x)  
            pp = self.pe(v)
            v = v.flatten(2).transpose(1, 2)

            if self.area > 1:
                qk = qk.reshape(B * self.area, N // self.area, C * 2) 
                v = v.reshape(B * self.area, N // self.area, C)
                B, N, _ = qk.shape
            q, k = qk.split([C, C], dim=2)  
            q = q.view(B, N, self.num_heads, self.head_dim)    
            k = k.view(B, N, self.num_heads, self.head_dim) 
            v = v.view(B, N, self.num_heads, self.head_dim)    
  
            x = flash_attn_func(    
                q.contiguous().half(),     
                k.contiguous().half(),
                v.contiguous().half()     
            ).to(q.dtype)

            if self.area > 1:
                x = x.reshape(B // self.area, N * self.area, C) 
                B, N, _ = x.shape
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:     
            qk = self.qk(x).flatten(2)
            v = self.v(x)
            pp = self.pe(v) 
            v = v.flatten(2)
            if self.area > 1:     
                qk = qk.reshape(B * self.area, C * 2, N // self.area)
                v = v.reshape(B * self.area, C, N // self.area)
                B, _, N = qk.shape
  
            q, k = qk.split([C, C], dim=1) 
            q = q.view(B, self.num_heads, self.head_dim, N)     
            k = k.view(B, self.num_heads, self.head_dim, N)
            v = v.view(B, self.num_heads, self.head_dim, N)
            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values    
            exp_attn = torch.exp(attn - max_attn)     
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))    
 
            if self.area > 1:
                x = x.reshape(B // self.area, C, N * self.area)  
                B, _, N = x.shape
            x = x.reshape(B, C, H, W)

        return self.proj(x + pp)   
  
     
class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.   
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while    
    maintaining effectiveness.    
    
    Attributes:
        attn (AAttn): Area-attention module for processing spatial features. 
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.   

    Methods:    
        _init_weights: Initializes module weights using truncated normal distribution.     
        forward: Applies area-attention and feed-forward processing to input tensor.   
  
    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32) 
        >>> output = block(x) 
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):     
        """
        Initializes an Area-attention block module for efficient feature extraction in YOLO models.

        This module implements an area-attention mechanism combined with a feed-forward network for processing feature
        maps. It uses a novel area-based attention approach that is more efficient than traditional self-attention
        while maintaining effectiveness.

        Args:  
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.     
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
     
        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)    
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
  
        self.apply(self._init_weights)

    def _init_weights(self, m):  
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):     
            nn.init.trunc_normal_(m.weight, std=0.02)  
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  

    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""  
        x = x + self.attn(x)    
        return x + self.mlp(x)
    

class A2C2f(nn.Module):    
    """   
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.   

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature  
    processing. It supports both area-attention and standard convolution modes.

    Attributes:  
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.   
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.    
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.  
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.
    
    Methods:   
        forward: Processes input through area-attention or standard convolution pathway.     

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32) 
        >>> output = m(x)
        >>> print(output.shape)    
        torch.Size([1, 512, 32, 32])     
    """ 
 
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        """
        Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.
 
        Args:     
            c1 (int): Number of input channels.
            c2 (int): Number of output channels. 
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.   
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.     
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.   
        """  
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."    

        self.cv1 = Conv(c1, c_, 1, 1)    
        self.cv2 = Conv((1 + n) * c_, c2, 1)   
     
        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None  
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g) 
            for _ in range(n)
        )
        # print(c1, c2, n, a2, area)

    def forward(self, x):     
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]  
        y.extend(m(y[-1]) for m in self.m) 
        y = self.cv2(torch.cat(y, 1))  
        if self.gamma is not None:  
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y     
        return y     
   
class C3_Block(nn.Module):
    """CSP Bottleneck with 3 convolutions.""" 

    def __init__(self, c1, c2, module=partial(Bottleneck, k=(1, 3), shortcut=True, e=0.5), n=1, e=0.5, selfatt=False):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)     
        self.cv2 = Conv(c1, c_, 1, 1) 
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2) 
        if selfatt:
            self.m = nn.Sequential(*(module(c_) for _ in range(n)))
        else:
            self.m = nn.Sequential(*(module(c_, c_) for _ in range(n)))   
   
    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""  
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)) 

class C2f_Block(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, module=partial(Bottleneck, k=(3, 3), shortcut=True, e=0.5), n=1, e=0.5, selfatt=False):   
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""    
        super().__init__() 
        self.c = int(c2 * e)  # hidden channels 
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)    
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        if selfatt:   
            self.m = nn.ModuleList(module(self.c) for _ in range(n))   
        else:  
            self.m = nn.ModuleList(module(self.c, self.c) for _ in range(n))   

    def forward(self, x):     
        """Forward pass through C2f layer."""  
        y = list(self.cv1(x).chunk(2, 1))    
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))   
  
class C3k_Block(nn.Module):  
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""  
     
    def __init__(self, c1, c2, module=partial(Bottleneck, k=(3, 3), shortcut=True, e=1.0), n=1, e=0.5, selfatt=False):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels     
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)     
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        if selfatt:
            self.m = nn.Sequential(*(module(c_) for _ in range(n)))
        else: 
            self.m = nn.Sequential(*(module(c_, c_) for _ in range(n))) 

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""    
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k2_Block(nn.Module):   
    def __init__(self, c1, c2, module=partial(Bottleneck, k=(3, 3), shortcut=True, e=0.5), n=1, c3k=True, e=0.5, selfatt=False):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1) 
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        if selfatt:   
            self.m = nn.ModuleList(
                    C3k_Block(self.c, self.c, module, 2, selfatt=selfatt) if c3k else module(self.c) for _ in range(n)     
                )  
        else:     
            self.m = nn.ModuleList(
                    C3k_Block(self.c, self.c, module, 2) if c3k else module(self.c, self.c) for _ in range(n)     
                )
    
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)     
        return self.cv2(torch.cat(y, 1))

class Scale(nn.Module):    
    """
    Scale vector by element multiplications.
    """  
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()   
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)   

    def forward(self, x):
        return x * self.scale    

class StarReLU(nn.Module):  
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()     
        self.inplace = inplace    
        self.relu = nn.ReLU(inplace=inplace)   
        self.scale = nn.Parameter(scale_value * torch.ones(1),   
            requires_grad=scale_learnable)    
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)   
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

class Mlp(nn.Module):     
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks. 
    Mostly copied from timm.   
    """   
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=StarReLU, drop=0., mlp_ratio=4, bias=False, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)
 
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=bias)   
        self.drop2 = nn.Dropout(drop_probs[1]) 
  
    def forward(self, x):
        x = self.fc1(x)  
        x = self.act(x)
        x = self.drop1(x)     
        x = self.fc2(x) 
        x = self.drop2(x)
        return x  

class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.

    Args:    
        affine_shape (int, list or tuple): The shape of affine weight and bias.    
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,   
            the affine_shape is the same as normalized_dim by default. 
            To adapt to different situations, we offer this argument here.   
        normalized_dim (tuple or list): Which dims to compute mean and variance.    
        scale (bool): Flag indicates whether to use scale or not.   
        bias (bool): Flag indicates whether to use scale or not.
   
        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True; 
            For input shape of (B, C, H, W),    
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),   
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.
  
        For the several metaformer baslines,   
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).     
    """
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale   
        self.use_bias = bias 
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps    
 
    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True) 
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)     
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:    
            x = x * self.weight   
        if self.use_bias:
            x = x + self.bias
        return x

class MetaFormer_Block(nn.Module):     
    """
    Implementation of one MetaFormer block. 
    """
    def __init__(self, in_dim, dim,
                 token_mixer=nn.Identity, mlp=Mlp, 
                 norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6), 
                 drop_path=0., mlp_ratio=2, 
                 layer_scale_init_value=None, res_scale_init_value=None, selfatt=False
                 ):

        super().__init__()

        self.norm1 = norm_layer((dim, 1, 1))
        if selfatt:  
            self.token_mixer = token_mixer(dim)
        else:  
            self.token_mixer = token_mixer(dim, dim) 
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()     
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
  
        self.norm2 = norm_layer((dim, 1, 1))
        self.mlp = mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity() 
  
        self.conv1x1 = Conv(in_dim, dim, 1) if in_dim != dim else nn.Identity()    
        
    def forward(self, x):
        x = self.conv1x1(x)
        # x size: [B, C, H, W]
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(     
                    self.token_mixer(self.norm1(x)) 
                )    
            ) 
        x = self.res_scale2(x) + \
            self.layer_scale2(   
                self.drop_path2(     
                    self.mlp(self.norm2(x))  
                )   
            )
        return x

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()  
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps) 
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
     
class MonaOp(nn.Module):  
    def __init__(self, in_features):    
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)    
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
    
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )  
 
    def forward(self, x):
        identity = x     
        conv1_x = self.conv1(x)     
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
  
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
    
        identity = x
   
        x = self.projector(x)
    
        return identity + x     

class Mona(nn.Module): 
    def __init__(self,     
                 in_dim):   
        super().__init__()  
    
        self.project1 = nn.Conv2d(in_dim, 64, 1)    
        self.nonlinear = F.gelu
        self.project2 = nn.Conv2d(64, in_dim, 1) 
 
        self.dropout = nn.Dropout(p=0.1)
 
        self.adapter_conv = MonaOp(64)

        self.norm = LayerNorm2d(in_dim)    
        self.gamma = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim, 1, 1))
 
    def forward(self, x, hw_shapes=None): 
        identity = x
 
        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)   

        project1 = self.adapter_conv(project1)  

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)     
        project2 = self.project2(nonlinear)
  
        return identity + project2
     
class MetaFormer_Mona(nn.Module):
    """  
    Implementation of one MetaFormer block.    
    """
    def __init__(self, in_dim, dim,    
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6),
                 drop_path=0., mlp_ratio=2,
                 layer_scale_init_value=None, res_scale_init_value=None, selfatt=False
                 ):     
  
        super().__init__() 
 
        self.norm1 = norm_layer((dim, 1, 1))
        if selfatt:
            self.token_mixer = token_mixer(dim) 
        else:     
            self.token_mixer = token_mixer(dim, dim)    
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()   
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()  
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        self.mona1 = Mona(dim)   
 
        self.norm2 = norm_layer((dim, 1, 1))
        self.mlp = mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()    
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity() 
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()   
        self.mona2 = Mona(dim)     
        
        self.conv1x1 = Conv(in_dim, dim, 1) if in_dim != dim else nn.Identity()    
  
    def forward(self, x):
        x = self.conv1x1(x)    
        # x size: [B, C, H, W]
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.mona1(x)

        x = self.res_scale2(x) + \
            self.layer_scale2( 
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )   
            )
        x = self.mona2(x)
    
        return x 
     
class NCHW2NLC2NCHW(nn.Module):
    def __init__(self, dim, module): 
        super().__init__()   
   
        self.module = module(dim)
    
    def forward(self, x):
        B, C, H, W = x.size()   
        x_nlc = x.flatten(2).permute(0, 2, 1) # B C H W -> N L C   
        x_nlc = self.module(x_nlc) # 经过对应的需要NLC输入的模块 
        x_nchw = x_nlc.permute(0, 2, 1).view([B, C, H, W]).contiguous() # N L C -> B C H W   
        return x_nchw

class MetaFormer_SEFN(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, in_dim, dim,  
                 token_mixer=nn.Identity,
                 norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6),
                 drop_path=0., mlp_ratio=2,
                 layer_scale_init_value=None, res_scale_init_value=None, selfatt=False 
                 ):
   
        super().__init__()   

        self.norm1 = norm_layer((dim, 1, 1))
        if selfatt:    
            self.token_mixer = token_mixer(dim)
        else:     
            self.token_mixer = token_mixer(dim, dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
   
        self.norm2 = norm_layer((dim, 1, 1))   
        self.mlp = SEFN(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)  
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()    
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity() 
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()     
        
        self.conv1x1 = Conv(in_dim, dim, 1) if in_dim != dim else nn.Identity()     
        
    def forward(self, x):
        x = self.conv1x1(x)    
        x_spatial = x 
        # x size: [B, C, H, W]
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(     
                    self.token_mixer(self.norm1(x))
                )
            )   
        x = self.res_scale2(x) + \
            self.layer_scale2(  
                self.drop_path2(
                    self.mlp(self.norm2(x), x_spatial)  
                )     
            )
        return x

class MetaFormer_Mona_SEFN(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, in_dim, dim,
                 token_mixer=nn.Identity,
                 norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6),
                 drop_path=0., mlp_ratio=2,
                 layer_scale_init_value=None, res_scale_init_value=None, selfatt=False
                 ): 
 
        super().__init__()   
     
        self.norm1 = norm_layer((dim, 1, 1))  
        if selfatt: 
            self.token_mixer = token_mixer(dim)    
        else:    
            self.token_mixer = token_mixer(dim, dim) 
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()     
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()  
        self.mona1 = Mona(dim)

        self.norm2 = norm_layer((dim, 1, 1)) 
        self.mlp = SEFN(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim) 
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()    
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        self.mona2 = Mona(dim)
        
        self.conv1x1 = Conv(in_dim, dim, 1) if in_dim != dim else nn.Identity()
        
    def forward(self, x):  
        x = self.conv1x1(x)
        x_spatial = x
        # x size: [B, C, H, W]
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))    
                )
            ) 
        x = self.mona1(x) 

        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(   
                    self.mlp(self.norm2(x), x_spatial)
                )
            )
        x = self.mona2(x)     

        return x    
