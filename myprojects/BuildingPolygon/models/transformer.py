import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """
    用于将图片切块，映射到线性嵌入的模块。
    输入：B x C x H x W
    输出：B x N x D
    - B: batch_size
    - C: 通道数
    - H, W: 图像高宽
    - N: 分块数 (H/P * W/P)
    - D: 嵌入维度
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 平方滤波器实现切块与线性映射
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置编码
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        # 检查图像大小是否可以被切分
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image size must be divisible by patch size."
        
        # 切割+线性变换+展平 -> [B, D, H//P, W//P] -> [B, N, D]
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, D, N] -> [B, N, D]
        
        # 加上位置编码
        x = x + self.pos_embed
        return x

class PatchCosineEmbedding(nn.Module):
    """
    用于将图片切块，映射到线性嵌入的模块。
    输入：B x C x H x W
    输出：B x N x D
    - B: batch_size
    - C: 通道数
    - H, W: 图像高宽
    - N: 分块数 (H/P * W/P)
    - D: 嵌入维度
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 平方滤波器实现切块与线性映射
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置编码
        num_patches = (img_size // patch_size) ** 2

        position = torch.arange(num_patches).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(num_patches, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维使用 cos
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        # 检查图像大小是否可以被切分
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image size must be divisible by patch size."
        
        # 切割+线性变换+展平 -> [B, D, H//P, W//P] -> [B, N, D]
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, D, N] -> [B, N, D]
        
        # 加上位置编码
        x = x + self.pe.to(x.device)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block，包含：
    - 多头自注意力
    - MLP (前馈) 层
    - Layer Norm 和残差连接
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 多头自注意力
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # MLP 用于前馈网络：线性 -> 激活 -> Dropout -> 线性
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

        # 残差 + LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v, pad_mask=None):
        # x: [N, B, D] （注意：Transformer 的输入需要是 [sequence_length, batch_size, embedding_dim]）
        
        # 多头注意力
        q = self.norm1(q + self.attn(q, k, v, need_weights=False, key_padding_mask=pad_mask)[0])  # Residual + Norm1
        # 前馈网络
        q = self.norm2(q + self.mlp(q))  # Residual + Norm2
        return q


class VisionEncoder(nn.Module):
    """
    整体 Vision Transformer (ViT) 模型
    """

    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=3, 
                 embed_dim=256, 
                 num_heads=8, 
                 depth=12,  # Transformer Encoder 的层数
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super(VisionEncoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Patch Embedding 层
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Transformer Encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])


    def forward(self, x):
        # Patch Embedding 输出：[B, N, D]
        x = self.patch_embed(x)  # Patch Embedding
        x = x.transpose(0, 1)
        for layer in self.encoder:
            x = layer(x,x,x)
        
        return x

class PolygonDecoder(nn.Module):
    """
    
    """

    def __init__(self, 
                 out_channels=2, 
                 embed_dim=256, 
                 num_heads=8, 
                 depth=12,  # Transformer Encoder 的层数
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super(PolygonDecoder, self).__init__()

        self.coord_conv = nn.Conv1d(2,256,3,1,1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        # Transformer Encoder
        self.decoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_channels)  # 全连接，用于分类

    def forward(self, coords, img_embed, pad_mask):
        # Patch Embedding 输出：[B, N, D]
        coords = self.coord_conv(coords)  # B D N
        coords = coords.permute(2,0,1)
        # print(coords.shape)

        # pad_mask = pad_mask.repeat(self.num_heads,1,self.embed_dim//pad_mask.shape[-1])

        for layer in self.decoder:
            coords = layer(coords,img_embed,img_embed)
        coords = self.head(self.norm(coords))
        
        return coords
    
class PolygonDecoderV2(nn.Module):
    """
    
    """

    def __init__(self, 
                 embed_dim=256, 
                 num_heads=8, 
                 depth=12,  # Transformer Encoder 的层数
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super(PolygonDecoderV2, self).__init__()

        # self.coord_conv = nn.Conv1d(2,256,3,1,1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        # Transformer Encoder
        self.decoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # 分类头
        # self.norm = nn.LayerNorm(embed_dim)
        # self.head = nn.Linear(embed_dim, out_channels)  # 全连接，用于分类

    def forward(self, coords, img_embed):
        # Patch Embedding 输出：[B, N, D]
        # coords = self.coord_conv(coords)  # B D N
        # coords = coords.permute(2,0,1)
        # print(coords.shape)

        # pad_mask = pad_mask.repeat(self.num_heads,1,self.embed_dim//pad_mask.shape[-1])

        for layer in self.decoder:
            coords = layer(coords,img_embed,img_embed)
        # coords = self.head(self.norm(coords))
        
        return coords

class SeqEncoder(nn.Module):
    """
    
    """

    def __init__(self, 
                 in_channels=2, 
                 embed_dim=256, 
                 num_heads=8, 
                 depth=3,  # Transformer Encoder 的层数
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super(SeqEncoder, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_channels, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU()
        # Seq Embedding
        # Transformer Encoder
        # self.encoder = nn.ModuleList([
        #     TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        # ])


    def forward(self, coords, pad_mask):
        # coord 输入 # B N 2
        coords = self.proj(coords)  # B N D
        coords = self.relu(self.norm(coords.permute(0,2,1))).permute(2,0,1) # N B D
        # print(coords.shape)

        # pad_mask = pad_mask[:,:,0] # B N

        # for layer in self.encoder:
        #     coords = layer(coords,coords,coords,pad_mask)
        # coords = self.head(self.norm(coords))
        
        return coords

# 示例用法：
vit_model = VisionEncoder(
    img_size=224, 
    patch_size=16, 
    in_channels=3, 
    embed_dim=256, 
    num_heads=8, 
    depth=6, 
    mlp_ratio=4, 
    dropout=0.1
)

poly_model = PolygonDecoder(
    out_channels=2, 
    embed_dim=256, 
    num_heads=8, 
    depth=6, 
    mlp_ratio=4, 
    dropout=0.1
)


if __name__=='__main__':
    dummy_input = torch.randn(2, 3, 224, 224)  # 假设 batch_size=2, 图像大小为 224x224
    output = vit_model(dummy_input)
    print(output.shape)  # 期望输出：[2, 1000]，对应 2 张图片的分类输出

    dummy_query = torch.randn(2, 2, 20)
    output2 = poly_model(dummy_query, output)
    print(output2.shape)