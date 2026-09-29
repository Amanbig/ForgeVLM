import torch
import torch.nn as nn
import torch.nn.functional as F


def slice_into_tiles(image: torch.Tensor, tile_size: int = 32):
    """Slice high-res image (C, H, W) into a grid of (tile_h, tile_w) tiles.

    Returns:
        tiles: (num_tiles, C, tile_size, tile_size)
        (grid_h, grid_w): number of tiles along height and width
    """
    c, h, w = image.shape
    # Pad to multiple of tile_size if needed
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        _, h, w = image.shape

    grid_h = h // tile_size
    grid_w = w // tile_size

    # Extract non-overlapping tiles
    tiles = []
    for i in range(grid_h):
        for j in range(grid_w):
            tile = image[:, i * tile_size : (i + 1) * tile_size, j * tile_size : (j + 1) * tile_size]
            tiles.append(tile)

    return torch.stack(tiles, dim=0), (grid_h, grid_w)


class AnyResProcessor(nn.Module):
    """Dynamic High-Resolution / AnyRes Image Processor (LLaVA-NeXT style).

    Preserves aspect ratios and high detail by:
    1. Producing a downsampled global overview thumbnail.
    2. Slicing the full image into a grid of local high-res tiles.
    3. Encoding all tiles through the vision encoder.
    4. Inserting learned newline tokens at the end of each spatial patch row,
       so the language model retains 2D spatial awareness.
    """

    def __init__(self, embed_dim: int, tile_size: int = 32, patch_size: int = 8):
        super().__init__()
        self.tile_size = tile_size
        self.patch_size = patch_size
        self.patches_per_tile_dim = tile_size // patch_size
        self.image_newline = nn.Parameter(torch.randn(embed_dim) * 0.02)

    def arrange_tiles_with_newlines(
        self,
        local_tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """Arranges flattened tile tokens into a coherent 2D image with newline delimiters.

        Args:
            local_tokens: (grid_h * grid_w, patches_per_tile, embed_dim)
        Returns:
            (total_tokens_with_newlines, embed_dim)
        """
        p = self.patches_per_tile_dim
        d = local_tokens.size(-1)

        # Reshape to (grid_h, grid_w, p, p, d)
        tokens_2d = local_tokens.view(grid_h, grid_w, p, p, d)
        # Permute to full spatial grid: (grid_h * p, grid_w * p, d)
        tokens_full = tokens_2d.permute(0, 2, 1, 3, 4).contiguous().view(grid_h * p, grid_w * p, d)

        # Append newline token at the end of each row
        newline = self.image_newline.view(1, 1, d).expand(grid_h * p, 1, -1)
        with_newlines = torch.cat((tokens_full, newline), dim=1)  # (H, W+1, d)

        return with_newlines.view(-1, d)
