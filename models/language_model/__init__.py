import os
from .bert import BERT, Roberta
from .lstm import RNNEncoder

def build_text_encoder(cfg):
    if cfg.MODEL.USE_LSTM:
        language_encoder = RNNEncoder(
            cfg.GLOVE_DIR,
            cfg.MODEL.LSTM.HIDDEN_SIZE // 2 if cfg.MODEL.LSTM.BIDIRECTIONAL \
                else cfg.MODE.LSTM.HIDDEN_SIZE,
            cfg.MODEL.LSTM.BIDIRECTIONAL,
            cfg.MODEL.LSTM.DROPOUT,
            cfg.MODEL.LSTM_NUM_LAYERS,
            cfg.MODEL.LSTM.NAME
        )
    else:
        language_encoder = Roberta(
            cfg.MODEL.TEXT_MODEL.NAME,
            cfg.MODEL.CG.HIDDEN,
            cfg.MODEL.TEXT_MODEL.FREEZE
        )
    return language_encoder