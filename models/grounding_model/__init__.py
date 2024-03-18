from .modal_encoder import CrossModalEncoder
from .query_decoder import QueryDecoder
    
def build_encoder(cfg):
    return CrossModalEncoder(cfg)

def build_decoder(cfg):
    return QueryDecoder(cfg)