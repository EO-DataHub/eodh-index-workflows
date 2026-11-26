"""Code shared by the products: the input item, the AOI, the band reader, the brightness rule and the
single-index product."""

from .aoi import parse_aoi, write_aoi_cog
from .bands import band, brightness
from .fetch import fetch_item, item_input
from .index import reflectance, run_index_product

__all__ = [
    "band",
    "brightness",
    "fetch_item",
    "item_input",
    "parse_aoi",
    "reflectance",
    "run_index_product",
    "write_aoi_cog",
]
