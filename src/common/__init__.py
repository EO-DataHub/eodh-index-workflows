"""Code shared by the products: the AOI, the band reader and the brightness rule."""

from .aoi import parse_aoi, write_aoi_cog
from .bands import band, brightness, require_item

__all__ = ["band", "brightness", "parse_aoi", "require_item", "write_aoi_cog"]
