from .models.mlp.grid_cache import GridCache as MLPGridCache
from .models.mdn_v2.grid_cache import GridCache as MDNV2GridCache


MLPGridCache.verify_and_generate_cache(device='cpu')
MDNV2GridCache.verify_and_generate_cache()
