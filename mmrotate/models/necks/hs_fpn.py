import torch
import torch.nn as nn
import torch.nn.functional as F


def dct_highpass_filter(x, alpha):
    """
    使用2D DCT（通过FFT近似）实现高通滤波。
    输入x: [B, C, H, W]
    alpha: 控制保留的低频比例，取值[0,1]
    返回高频响应 F (与x形状相同)
    """
    B, C, H, W = x.shape
    # 使用rFFT变换到频域（实数FFT，节省计算）
    # 注意：rFFT输出形状为 [B, C, H, W//2+1]
    x_fft = torch.fft.rfft2(x, norm='ortho')

    # 创建高通掩码
    # 生成频率坐标
    freq_h = torch.fft.fftfreq(H, device=x.device).view(-1, 1).abs()  # [H,1]
    freq_w = torch.fft.rfftfreq(W, device=x.device).view(1, -1)  # [1, W//2+1]
    # 归一化到[0,1]
    freq_h = freq_h / freq_h.max()
    freq_w = freq_w / freq_w.max()
    # 计算截止半径：低频区域为alpha比例，即保留半径 < alpha 的部分为0
    mask = ((freq_h ** 2 + freq_w ** 2) ** 0.5) >= alpha  # True表示高频区域（保留），False表示低频（滤除）
    mask = mask.float().to(x.device)  # [H, W//2+1]

    # 应用掩码
    x_fft_filtered = x_fft * mask.unsqueeze(0).unsqueeze(0)

    # 逆变换回空间域
    x_filtered = torch.fft.irfft2(x_fft_filtered, s=(H, W), norm='ortho')
    return x_filtered


class HFP(nn.Module):
    """
    高频感知模块 (High Frequency Perception Module)
    包含：高通滤波器、通道路径(CP)、空间路径(SP)
    """

    def __init__(self, channels, alpha=0.25, k=16):
        super().__init__()
        self.alpha = alpha
        self.k = k  # 池化后保留的尺寸，论文中设为16

        # 通道路径 (CP)
        self.gap_pool = nn.AdaptiveAvgPool2d((k, k))
        self.gmp_pool = nn.AdaptiveMaxPool2d((k, k))
        self.relu = nn.ReLU(inplace=True)
        # 两个1x1分组卷积，组数暂时设为与通道数相同（深度卷积）？论文说1x1 group convolution，我们设groups=channels（即深度卷积）
        self.conv_cp1 = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        self.conv_cp2 = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        self.conv_cp3 = nn.Conv2d(channels * 2, channels, kernel_size=1)  # 拼接后融合

        # 空间路径 (SP)
        self.conv_sp = nn.Conv2d(channels, 1, kernel_size=1)

        # 输出卷积
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, C, H, W]
        # 高通滤波得到高频响应 F
        F = dct_highpass_filter(x, self.alpha)

        # ----- 通道路径 (CP) -----
        # 对F进行GAP和GMP，得到 [B, C, k, k]
        gap = self.gap_pool(F)
        gmp = self.gmp_pool(F)
        # ReLU
        gap = self.relu(gap)
        gmp = self.relu(gmp)
        # 对每个通道求和，压缩成 [B, C, 1, 1]（通过全局平均/最大？这里论文说"sum across each channel to generate two one-dimensional feature vectors"，可能是在k*k维度上求和）
        # 我们理解为对k*k空间维度求和，得到 [B, C]
        gap = gap.sum(dim=(2, 3))  # [B, C]
        gmp = gmp.sum(dim=(2, 3))  # [B, C]
        # 重塑为 [B, C, 1, 1]
        gap = gap.view(gap.size(0), gap.size(1), 1, 1)
        gmp = gmp.view(gmp.size(0), gmp.size(1), 1, 1)

        # 分别通过1x1分组卷积
        score_gap = self.conv_cp1(gap)  # [B, C, 1, 1]
        score_gmp = self.conv_cp2(gmp)  # [B, C, 1, 1]
        # 沿通道拼接
        cat_scores = torch.cat([score_gap, score_gmp], dim=1)  # [B, 2C, 1, 1]
        # 融合得到最终通道权重
        u_cp = self.conv_cp3(cat_scores)  # [B, C, 1, 1]
        u_cp = torch.sigmoid(u_cp)  # 注意：论文未明确激活函数，但通常注意力用sigmoid

        # 应用通道权重
        x_cp = x * u_cp

        # ----- 空间路径 (SP) -----
        u_sp = self.conv_sp(F)  # [B, 1, H, W]
        u_sp = torch.sigmoid(u_sp)
        x_sp = x * u_sp

        # 融合CP和SP的输出
        out = x_cp + x_sp
        out = self.out_conv(out)
        return out


class SDP(nn.Module):
    """
    空间依赖感知模块 (Spatial Dependency Perception Module)
    在上下层特征之间进行像素级交叉注意力
    """

    def __init__(self, channels, base_size):  # base_size 用于分块，应等于 C5 的尺寸（如7x7）
        super().__init__()
        self.channels = channels
        self.base_h, self.base_w = base_size  # 例如(7,7)
        # 生成Q, K, V的1x1卷积
        self.conv_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1)
        # 缩放因子
        self.scale = channels ** 0.5

    def forward(self, c_i, p_i1):
        """
        c_i: 下层特征 [B, C, H_i, W_i]
        p_i1: 上层特征（已上采样到与c_i相同尺寸） [B, C, H_i, W_i]
        """
        B, C, H, W = c_i.shape
        # 生成Q, K, V
        Q = self.conv_q(c_i)  # [B, C, H, W]
        K = self.conv_k(p_i1)  # [B, C, H, W]
        V = self.conv_v(p_i1)  # [B, C, H, W]

        # 计算分块数量
        n_h = H // self.base_h
        n_w = W // self.base_w
        # 确保整除（假设输入尺寸是base_size的整数倍，否则需要padding或调整）
        assert H % self.base_h == 0 and W % self.base_w == 0, "输入尺寸必须能被base_size整除"

        # 将特征划分为 n = n_h * n_w 个块，每个块形状 [B, C, base_h, base_w]
        # 使用unfold操作或reshape
        Q_blocks = Q.view(B, C, n_h, self.base_h, n_w, self.base_w).permute(0, 2, 4, 1, 3, 5).contiguous()
        Q_blocks = Q_blocks.view(B, n_h * n_w, C, self.base_h * self.base_w)  # [B, n, C, L] where L = base_h*base_w
        K_blocks = K.view(B, C, n_h, self.base_h, n_w, self.base_w).permute(0, 2, 4, 1, 3, 5).contiguous()
        K_blocks = K_blocks.view(B, n_h * n_w, C, self.base_h * self.base_w)
        V_blocks = V.view(B, C, n_h, self.base_h, n_w, self.base_w).permute(0, 2, 4, 1, 3, 5).contiguous()
        V_blocks = V_blocks.view(B, n_h * n_w, C, self.base_h * self.base_w)

        # 在每个块内计算交叉注意力
        # Q_blocks, K_blocks: [B, n, C, L], 我们转置后计算
        Q_blocks = Q_blocks.transpose(-2, -1)  # [B, n, L, C]
        K_blocks = K_blocks.transpose(-2, -1)  # [B, n, L, C]
        V_blocks = V_blocks.transpose(-2, -1)  # [B, n, L, C]

        # 计算相似度矩阵: [B, n, L, L]
        attn = torch.matmul(Q_blocks, K_blocks.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)

        # 加权求和
        out_blocks = torch.matmul(attn, V_blocks)  # [B, n, L, C]
        out_blocks = out_blocks.transpose(-2, -1)  # [B, n, C, L]

        # 重塑回原始形状
        out_blocks = out_blocks.view(B, n_h, n_w, C, self.base_h, self.base_w)
        out_blocks = out_blocks.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B, C, n_h, base_h, n_w, base_w]
        out = out_blocks.view(B, C, H, W)

        # 残差连接
        return c_i + out


class HSFPN(nn.Module):
    """
    HS-FPN整体结构
    输入: [c2, c3, c4, c5] (来自骨干网络的四个特征图，通道数已通过1x1卷积统一到256)
    输出: [p2, p3, p4, p5] (增强后的特征金字塔)
    """

    def __init__(self, in_channels=256, alpha=0.25, k=16):
        super().__init__()
        self.in_channels = in_channels
        self.alpha = alpha
        self.k = k

        # HFP模块用于每个侧向连接（但论文说所有侧向连接都包含HFP）
        self.hfp_c2 = HFP(in_channels, alpha, k)
        self.hfp_c3 = HFP(in_channels, alpha, k)
        self.hfp_c4 = HFP(in_channels, alpha, k)
        self.hfp_c5 = HFP(in_channels, alpha, k)

        # SDP模块用于P2,P3,P4（需要知道base_size，这里假设输入尺寸固定，或动态计算）
        # 由于输入尺寸可变，我们可以在forward中根据c5的尺寸动态设置base_size
        # 这里先不实例化SDP，而是在forward中根据实际尺寸动态创建，或者使用一个通用模块，在forward中计算分块
        # 简单起见，我们在forward中动态构建SDP（但这样会创建新的模块，可能影响梯度）
        # 更好的方式是设计SDP使其能自适应不同尺寸，我们可以将base_size作为参数传入forward
        # 为了简化，我们假设输入尺寸是预定义的，或者我们使用一个可处理任意尺寸的SDP实现（如通过插值调整base_size）
        # 我们将在forward中动态创建SDP实例，但为了效率，可以用一个函数来实现SDP

    def forward(self, feats):
        """
        feats: [c2, c3, c4, c5] 每个都是 [B, C, H, W] 且C=256
        """
        c2, c3, c4, c5 = feats

        # 先应用HFP增强每个侧向特征
        c2_hfp = self.hfp_c2(c2)
        c3_hfp = self.hfp_c3(c3)
        c4_hfp = self.hfp_c4(c4)
        c5_hfp = self.hfp_c5(c5)

        # 构建金字塔：自顶向下融合
        # P5 = c5_hfp
        p5 = c5_hfp

        # P4: c4_hfp + 上采样P5
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode='nearest')  # 最近邻上采样（论文可能用双线性，但没指定）
        p4 = c4_hfp + p5_up
        # 应用SDP在c4_hfp和p5_up之间（注意SDP输入是原始c4_hfp和上采样的p5_up，还是c4_hfp和p5_up？论文说SDP输入是{C_i, P_{i+1}}，其中P_{i+1}已上采样）
        sdp_out = self._apply_sdp(c4_hfp, p5_up)  # 自定义函数实现SDP
        p4 = p4 + sdp_out  # 注意：论文中SDP的输出是加回C_i，这里我们加在p4上？论文表述：SDP输出是经过交叉注意力的新特征，然后加到C_i上。这里C_i是c4_hfp，我们已经在p4中包含了c4_hfp，但为了保留残差连接，我们将SDP的输出加到p4上（等同于加到c4_hfp然后加上p5_up，但p5_up也会受SDP影响？）

        # 更准确的：SDP的输出应该直接加到c4_hfp上，然后再与p5_up相加。但代码中我们可以先计算sdp_out加到c4_hfp，再与p5_up相加。
        # 我们重新整理：
        # sdp_out = SDP(c4_hfp, p5_up)
        # c4_hfp_enhanced = c4_hfp + sdp_out
        # p4 = c4_hfp_enhanced + p5_up
        # 这样符合论文描述。我们按照这个逻辑实现。

        # 为了代码清晰，我们写一个辅助函数处理每个层
        def apply_sdp_and_merge(low_feat, high_feat):
            sdp_out = self._apply_sdp(low_feat, high_feat)
            low_enhanced = low_feat + sdp_out
            return low_enhanced

        # P4
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        c4_enhanced = apply_sdp_and_merge(c4_hfp, p5_up)
        p4 = c4_enhanced + p5_up

        # P3
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        c3_enhanced = apply_sdp_and_merge(c3_hfp, p4_up)
        p3 = c3_enhanced + p4_up

        # P2
        p3_up = F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        c2_enhanced = apply_sdp_and_merge(c2_hfp, p3_up)
        p2 = c2_enhanced + p3_up

        # 最后每个层经过3x3卷积（论文中提到）
        # 我们可以定义额外的卷积，但为了简化，假设外部已经做了，或者我们在这里加
        # 通常在FPN后接3x3卷积，我们在外部处理或在这里定义
        # 我们在__init__中定义输出卷积
        # 简单起见，我们在类中定义这些输出卷积
        # 但为了不使代码过长，我们假设用户会自行添加输出卷积

        return [p2, p3, p4, p5]

    def _apply_sdp(self, low, high):
        """ 应用SDP模块。由于尺寸可变，我们动态构建SDP所需的参数。"""
        B, C, H, W = low.shape
        # 以C5的尺寸作为base_size，即从high中推断？但high是上采样后的，尺寸与low相同。
        # 论文中base_size应等于C5的尺寸（即最低分辨率特征的尺寸）。我们需要传入原始C5的尺寸，或者从外部获取。
        # 这里我们假设外部会传入base_h, base_w，或者在初始化时固定。
        # 为了简化，我们在这里使用固定base_size为7（假设C5为7x7），但实际可能变化。
        # 更好的方式：在forward时传入base_size。
        # 我们可以在HSFPN初始化时保存base_size（假设输入固定），或动态计算。
        # 为了通用性，我们实现一个动态SDP，其中base_size取H和W的某一固定比例？但论文明确说使用C5的尺寸作为块大小。
        # 因此，我们在forward中需要知道C5的原始尺寸（未上采样前的）。我们在HSFPN中保存c5的尺寸作为base_size。
        # 修改HSFPN，在第一次forward时记录c5的尺寸，作为base_size。
        if not hasattr(self, 'base_h') or not hasattr(self, 'base_w'):
            # 从c5获取，但c5是输入之一，我们需要在forward中传入。可以在HSFPN之外计算，然后作为参数传入。
            # 这里我们设计为在forward时传入base_size，或者从c5的尺寸获取。
            # 为了简化，我们假设输入尺寸固定，并且在初始化时指定base_size。
            # 如果输入可变，可以使用adaptive pooling将high特征池化到固定尺寸？但论文没有。
            # 我们这里就假设base_size是已知的，并在初始化时设置。
            pass
        # 由于HSFPN没有保存base_size，我们无法在这里获得。我们可以在HSFPN初始化时传入一个base_size参数，或者在forward时额外传入。
        # 我们修改HSFPN的__init__，增加base_size参数。
        # 为了当前函数能工作，我们临时从low的尺寸推断base_size？不合理。
        # 我们决定：在HSFPN中增加一个属性self.base_size，并在forward时使用。
        # 下面的代码依赖self.base_h, self.base_w，我们将在HSFPN初始化时设置。
        # 我们重写HSFPN如下：


# 修改后的HSFPN，包含base_size参数
class HSFPN(nn.Module):
    def __init__(self, in_channels=256, alpha=0.25, k=16, base_size=(7, 7)):
        super().__init__()
        self.in_channels = in_channels
        self.alpha = alpha
        self.k = k
        self.base_h, self.base_w = base_size

        # HFP模块
        self.hfp_c2 = HFP(in_channels, alpha, k)
        self.hfp_c3 = HFP(in_channels, alpha, k)
        self.hfp_c4 = HFP(in_channels, alpha, k)
        self.hfp_c5 = HFP(in_channels, alpha, k)

        # SDP模块（我们实例化SDP，但SDP需要base_size，每个层共享一个？还是独立？论文中每个侧向连接都有一个SDP，但参数可能共享？不共享更好）
        # 我们为每个需要SDP的层创建独立的SDP实例，但每个SDP需要知道base_size。
        self.sdp_p4 = SDP(in_channels, base_size)
        self.sdp_p3 = SDP(in_channels, base_size)
        self.sdp_p2 = SDP(in_channels, base_size)

        # 输出卷积（可选，为了符合FPN结构）
        self.conv_p2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_p3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_p4 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_p5 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, feats):
        c2, c3, c4, c5 = feats
        # 应用HFP
        c2_hfp = self.hfp_c2(c2)
        c3_hfp = self.hfp_c3(c3)
        c4_hfp = self.hfp_c4(c4)
        c5_hfp = self.hfp_c5(c5)

        # P5
        p5 = c5_hfp

        # P4
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        sdp_out = self.sdp_p4(c4_hfp, p5_up)  # SDP输出加到low上
        c4_enhanced = c4_hfp + sdp_out
        p4 = c4_enhanced + p5_up

        # P3
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        sdp_out = self.sdp_p3(c3_hfp, p4_up)
        c3_enhanced = c3_hfp + sdp_out
        p3 = c3_enhanced + p4_up

        # P2
        p3_up = F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        sdp_out = self.sdp_p2(c2_hfp, p3_up)
        c2_enhanced = c2_hfp + sdp_out
        p2 = c2_enhanced + p3_up

        # 输出卷积
        p2 = self.conv_p2(p2)
        p3 = self.conv_p3(p3)
        p4 = self.conv_p4(p4)
        p5 = self.conv_p5(p5)

        return [p2, p3, p4, p5]