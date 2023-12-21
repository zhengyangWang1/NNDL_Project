from model.encoder.resnet import ResNetEncoder
from model.encoder.transformer_encoder import TransformerEncoder
# TODO PackedCrossEntropyLoss实现应该移动

from model.decoder.gru import GRUDecoder, PackedCrossEntropyLoss
from model.decoder.transformer_decoder import TransformerDecoder

from model.model import CNNRNNStruct,CNNTransformerModel
