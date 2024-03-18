from cgitb import text
import torch
import torch.nn.functional as F

from torch import nn
from utils.video_list import NestedTensor

from pytorch_pretrained_bert.modeling import BertModel
from transformers import RobertaModel, RobertaTokenizerFast


class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, enc_num, pretrain_weight):
        super().__init__()
        if name == 'bert-base-uncased':
            self.num_channels = 768
        else:
            self.num_channels = 1024

        self.enc_num = enc_num
        self.bert = BertModel.from_pretrained(pretrain_weight)

        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        if self.enc_num > 0:
            all_encoder_layers, _ = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
            # use the output of the X-th transformer encoder layers
            xs = all_encoder_layers[self.enc_num - 1]
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out


class Roberta(nn.Module):
    def __init__(self, name, outdim, freeze=False) -> None:
        super().__init__()
        self.body = RobertaModel.from_pretrained("model_zoo/roberta-base/")
        self.tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name_or_path='model_zoo/roberta/')

        if freeze:
            for p in self.body.parameters():
                p.requires_grad_(False)

        config = self.body.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=outdim,
            dropout=0.1,
        )

    def forward(self, texts, device):
        tokenized = self.tokenizer.batch_encode_plus(texts, 
                        padding="longest", return_tensors="pt").to(device)
        encoded_text = self.body(**tokenized)
        text_cls = encoded_text.pooler_output
        
        # Transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = self.resizer(text_memory)
        text_cls_resized = self.resizer(text_cls)
        
        return (text_attention_mask, text_memory_resized, tokenized), text_cls_resized


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output