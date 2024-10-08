import torch
import torch.nn as nn
import torch.nn.functional as F


class LWReg(nn.Module):
    def __init__(self, in_channel, out_channel, base_channel):
        super().__init__()
        self.in_channel = in_channel  # 1
        self.out_channel = out_channel  # 2
        self.base_channel = base_channel  # default 8

        bias_opt = True

        # Initial convolution
        self.initial_conv = self.conv_block(4 * in_channel, base_channel * 8, bias=bias_opt)

        # Convolutional blocks
        self.down_conv1 = self.conv_block(base_channel * 8, base_channel * 8, kernel_size=3, stride=1, bias=bias_opt)
        self.down_conv2 = self.conv_block(base_channel * 8, base_channel * 8, kernel_size=3, stride=1, bias=bias_opt)
        self.mid_conv1 = self.conv_block(base_channel * 6 + 4, base_channel * 6, kernel_size=3, stride=1, bias=bias_opt)
        self.mid_conv2 = self.conv_block(base_channel * 6, base_channel * 6, kernel_size=3, stride=1, bias=bias_opt)
        self.up_conv1 = self.conv_block(base_channel * 4 + 4, base_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.up_conv2 = self.conv_block(base_channel * 4, base_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.final_conv1 = self.conv_block(
            base_channel * 2 + 2, base_channel * 2, kernel_size=3, stride=1, bias=bias_opt
        )
        self.final_conv2 = self.conv_block(base_channel * 2, base_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.output_conv = self.output_block(
            base_channel * 2, out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Upsampling layers
        self.upsample1 = self.upsample_block(base_channel * 8, base_channel * 6)
        self.upsample2 = self.upsample_block(base_channel * 6, base_channel * 4)
        self.upsample3 = self.upsample_block(base_channel * 4, base_channel * 2)

        # Pooling layers
        self.max_pool_2 = nn.MaxPool2d(2, stride=2)
        self.max_pool_4 = nn.MaxPool2d(4, stride=4)
        self.max_pool_8 = nn.MaxPool2d(8, stride=8)
        self.avg_pool_2 = nn.AvgPool2d(2, stride=2)
        self.avg_pool_4 = nn.AvgPool2d(4, stride=4)
        self.avg_pool_8 = nn.AvgPool2d(8, stride=8)

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(inplace=True),
        )

    def upsample_block(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
        )

    def output_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias), nn.Tanh()
        )

    def forward(self, fixed_image, moving_image):
        # Multi-scale feature extraction
        x_avg_8 = self.avg_pool_8(fixed_image)
        y_avg_8 = self.avg_pool_8(moving_image)
        x_max_8 = self.max_pool_8(fixed_image)
        y_max_8 = self.max_pool_8(moving_image)

        # Initial feature concatenation
        initial_features = torch.cat((x_avg_8, y_avg_8, x_max_8, y_max_8), 1)

        # Network operations
        d0 = self.initial_conv(initial_features)  # channel: base_channel * 8
        d0 = self.down_conv1(d0)  # channel: base_channel * 8
        d0 = self.down_conv2(d0)  # channel: base_channel * 8
        d1 = self.upsample1(d0)  # channel: base_channel * 6

        x_avg_4 = self.avg_pool_4(fixed_image)
        y_avg_4 = self.avg_pool_4(moving_image)
        x_max_4 = self.max_pool_4(fixed_image)
        y_max_4 = self.max_pool_4(moving_image)
        d1 = F.interpolate(d1, size=(x_avg_4.size(2), x_avg_4.size(3)), mode="bilinear", align_corners=False)
        d1 = torch.cat((d1, x_avg_4, y_avg_4, x_max_4, y_max_4), 1)  # channel: base_channel * 6 + 4

        d1 = self.mid_conv1(d1)  # channel: base_channel * 6
        d1 = self.mid_conv2(d1)  # channel: base_channel * 6
        d2 = self.upsample2(d1)  # channel: base_channel * 4

        x_avg_2 = self.avg_pool_2(fixed_image)
        y_avg_2 = self.avg_pool_2(moving_image)
        x_max_2 = self.max_pool_2(fixed_image)
        y_max_2 = self.max_pool_2(moving_image)
        d2 = F.interpolate(d2, size=(x_avg_2.size(2), x_avg_2.size(3)), mode="bilinear", align_corners=False)
        d2 = torch.cat((d2, x_avg_2, y_avg_2, x_max_2, y_max_2), 1)  # channel: base_channel * 4 + 4

        d2 = self.up_conv1(d2)  # channel: base_channel * 4
        d2 = self.up_conv2(d2)  # channel: base_channel * 4
        d3 = self.upsample3(d2)  # channel: base_channel * 2
        d3 = F.interpolate(d3, size=(fixed_image.size(2), fixed_image.size(3)), mode="bilinear", align_corners=False)
        d3 = torch.cat((d3, fixed_image, moving_image), 1)  # channel: base_channel * 2 + 2

        d3 = self.final_conv1(d3)  # channel: base_channel * 2
        d3 = self.final_conv2(d3)  # channel: base_channel * 2

        deformation_field = self.output_conv(d3)  # channel: 2

        return deformation_field


class SpatialTransform2D(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super().__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)
        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]  # y, x

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=False)


class DiffeomorphicTransform2D(nn.Module):
    def __init__(self, size, mode="bilinear", num_steps=7):
        super().__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

        self.mode = mode
        self.num_steps = num_steps

    def forward(self, src, velocity_field):
        # Integrate velocity field to obtain displacement field
        flow = self.integrate_velocity(velocity_field)

        # Apply displacement field
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]  # y, x

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=False)

    def integrate_velocity(self, velocity_field):
        # Scaling and squaring method
        flow = velocity_field / (2**self.num_steps)
        for _ in range(self.num_steps):
            x = self.grid + flow
            flow = flow + self.sample_flow(flow, x)
        return flow

    def sample_flow(self, flow, x):
        shape = flow.shape[2:]
        for i in range(len(shape)):
            x[:, i, ...] = 2 * (x[:, i, ...] / (shape[i] - 1) - 0.5)
        x = x.permute(0, 2, 3, 1)
        x = x[..., [1, 0]]
        return F.grid_sample(flow, x, mode=self.mode, align_corners=False)
