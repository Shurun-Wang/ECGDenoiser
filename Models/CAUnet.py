import torch
from torch import nn
from torch.nn import functional as F
from ptflops import get_model_complexity_info
from Models.triplet_attention import TripletAttention

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, kernel_size=(2, 2), stride=2)


def Downsample(dim):
    return nn.Conv2d(dim, dim, kernel_size=(2, 2), stride=2)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=[1, 2], keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2_Block(nn.Module):
    """ ConvNeXtV2 Block.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dwconv = nn.Conv2d(dim_in, dim_in, kernel_size=7, padding=3, groups=dim_in)  # depthwise conv
        self.norm = LayerNorm(dim_in, eps=1e-6)
        self.pwconv1 = nn.Linear(dim_in, 4 * dim_in)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim_in)
        self.pwconv2 = nn.Linear(4 * dim_in, dim_out)
        self.res_conv = nn.Conv2d(dim_in, dim_out, (1, 1)) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = x + self.res_conv(input)
        return x


class CAUnet(nn.Module):
    """ ConvNeXt-based attention Unet
    Args:
        chans (int): Number of mag channels. Default: 1
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
    """
    def __init__(self, in_channel=1, out_channel=1, base_dim=32, multi_dim=4):
        super(CAUnet, self).__init__()

        dims = []
        for i in range(multi_dim):
            dims.append(2**i*base_dim)
        self.init_conv = nn.Conv2d(in_channel, dims[0], (1, 1))

        dim_in_out = list(zip(dims[:-1], dims[1:]))
        # layers
        self.downs, self.ups = nn.ModuleList([]), nn.ModuleList([])

        num_resolutions = len(dim_in_out)
        for ind, (dim_in, dim_out) in enumerate(dim_in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList(
                [ConvNeXtV2_Block(dim_in, dim_in),
                 ConvNeXtV2_Block(dim_in, dim_out),
                 Downsample(dim_out) if not is_last else nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNeXtV2_Block(mid_dim, mid_dim)
        self.mid_block2 = ConvNeXtV2_Block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(dim_in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList(
                [TripletAttention(),
                 ConvNeXtV2_Block(dim_out * 2, dim_in),
                 ConvNeXtV2_Block(dim_in, dim_in),
                 Upsample(dim_in) if not is_last else nn.Identity()]))

        self.final_conv = nn.Conv2d(dims[0], out_channel, (1, 1))

    def forward(self, x):

        h = []
        # Encoder
        x = self.init_conv(x)
        for block1, block2, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x)
        x = self.mid_block2(x)

        # Decoder
        for TA, block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = TA(x)
            x = block1(x)
            x = block2(x)
            x = upsample(x)
        return self.final_conv(x)


if __name__ == "__main__":
    # mag data shape [batch, 1, 64, 32]
    from calflops import calculate_flops
    from torchsummary import summary
    import time
    x = torch.randn(1, 1, 64, 32)
    model = CAUnet(in_channel=1, out_channel=1, base_dim=48, multi_dim=3)
    # time1 = time.time()
    y = model(x)
    # print(time.time()-time1)
    summary(model, (1, 64, 32))
    # input_shape = (batch_size, 14, 300) # ninapro
    input_shape = (1, 1, 64, 32)  # grabmyo
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

    #
    # # model = ConvNeXtV2()
    # # print(model)
    # x = torch.randn(1, 1, 64, 32)
    # y = model(x)
    # # with torch.cuda.device(0):
    # macs, params = get_model_complexity_info(model, (1, 64, 32), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
