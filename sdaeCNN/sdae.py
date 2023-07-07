import math
from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
from typing import Callable, Iterable, Optional, Tuple, List
import torch
import torch.nn as nn


def build_units(
    dimensions: Iterable[int], activation: Optional[torch.nn.Module]
) -> List[torch.nn.Module]:
    """
    Given a list of dimensions and optional activation, return a list of units where each unit is a linear
    layer followed by an activation layer.

    :param dimensions: iterable of dimensions for the chain
    :param activation: activation layer to use e.g. nn.ReLU, set to None to disable
    :return: list of instances of Sequential
    """

    def single_unit(in_dimension: int, out_dimension: int) -> torch.nn.Module:
        unit = [("linear", nn.Linear(in_dimension, out_dimension))]
        if activation is not None:
            unit.append(("activation", activation))
        return nn.Sequential(OrderedDict(unit))

    return [
        single_unit(embedding_dimension, hidden_dimension)
        for embedding_dimension, hidden_dimension in sliding_window(2, dimensions)
    ]


def build_encoder_units(
        dimensions:Iterable[Iterable[int]],activation:Optional[torch.nn.Module]
)->List[torch.nn.Module]:
    def single_unit(channelList:Iterable[int])->torch.nn.Module:
        layers =[]
        for in_channel,out_channel in sliding_window(2,channelList):
            layers.append(nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(activation)
            layers.append(nn.AvgPool2d(kernel_size=2,stride=2,padding=1))
        return nn.Sequential(*layers)
    encoders = []
    for cL in dimensions:
        encoders.append(single_unit(cL))
    return encoders

def build_decoder_units(
        dimensions:Iterable[Iterable[int]],activation:Optional[torch.nn.Module]
)->List[torch.nn.Module]:
    def single_unit(channelList:Iterable[int],isOut:bool)->torch.nn.Module:
        layers =[]
        channelList = list(reversed(channelList))
        in_last = 0
        out_last = 0
        if isOut:
            in_last = channelList[-2]
            out_last = channelList[-1]
            channelList = channelList[:-1]
        for in_channel,out_channel in sliding_window(2,channelList):
            layers.append(nn.Upsample(scale_factor=2.0))
            layers.append(nn.ConvTranspose2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(activation)
        if in_last + out_last !=0 :
            layers.append(nn.Upsample(scale_factor=2.0))
            layers.append(nn.ConvTranspose2d(in_last, out_last, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_last))
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    decoders = []
    dimensions_list = list(dimensions)
    for cL in dimensions_list:
        decoders.append(single_unit(cL,cL == dimensions_list[-1]))
    return decoders

def default_initialise_weight_bias_(
    weight: torch.Tensor, bias: torch.Tensor, gain: float
) -> None:
    """
    Default function to initialise the weights in a the Linear units of the StackedDenoisingAutoEncoder.

    :param weight: weight Tensor of the Linear unit
    :param bias: bias Tensor of the Linear unit
    :param gain: gain for use in initialiser
    :return: None
    """
    nn.init.xavier_uniform_(weight, gain)
    nn.init.constant_(bias, 0)



class StackedDenoisingAutoEncoder(nn.Module):
    def __init__(
        self,
        dimensions: List[int],
        activation: torch.nn.Module = nn.ReLU(),
        # activation: torch.nn.Module = nn.LeakyReLU(),

            final_activation: Optional[torch.nn.Module] = nn.ReLU(),
        weight_init: Callable[
            [torch.Tensor, torch.Tensor, float], None
        ] = default_initialise_weight_bias_,
        gain: float = nn.init.calculate_gain("relu"),
    ):
        """
        Autoencoder composed of a symmetric decoder and encoder components accessible via the encoder and decoder
        attributes. The dimensions input is the list of dimensions occurring in a single stack
        e.g. [100, 10, 10, 5] will make the embedding_dimension 100 and the hidden dimension 5, with the
        autoencoder shape [100, 10, 10, 5, 10, 10, 100].

        :param dimensions: list of dimensions occurring in a single stack
        :param activation: activation layer to use for all but final activation, default torch.nn.ReLU
        :param final_activation: final activation layer to use, set to None to disable, default torch.nn.ReLU
        :param weight_init: function for initialising weight and bias via mutation, defaults to default_initialise_weight_bias_
        :param gain: gain parameter to pass to weight_init
        """
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.dimensions = dimensions
        self.embedding_dimension = dimensions[0]
        self.hidden_dimension = dimensions[-1]
        # construct the encoder
        encoder_units = build_encoder_units(self.dimensions, activation)
        # encoder_units = build_encoder_units(self.dimensions[:-1], activation)

        # encoder_units.extend(
        #     build_units([self.dimensions[-2], self.dimensions[-1]], None)
        # )
        self.encoder = nn.Sequential(*encoder_units)
        # construct the decoder
        # decoder_units = build_units(reversed(self.dimensions[1:]), activation)
        decoder_units = build_decoder_units(reversed(self.dimensions), activation)
        # decoder_units.extend(
        #     build_units([self.dimensions[1], self.dimensions[0]], final_activation)
        # )
        self.decoder = nn.Sequential(*decoder_units)
        # initialise the weights and biases in the layers
        self._initialize_weights(gain)
        # for layer in concat([self.encoder, self.decoder]):
        #     weight_init(layer[0].weight, layer[0].bias, gain)

    def get_stack(self, index: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Given an index which is in [0, len(self.dimensions) - 2] return the corresponding subautoencoder
        for layer-wise pretraining.

        :param index: subautoencoder index
        :return: tuple of encoder and decoder units
        """
        if (index > len(self.dimensions) - 1) or (index < 0):
            raise ValueError(
                "Requested subautoencoder cannot be constructed, index out of range."
            )
        return self.encoder[index], self.decoder[-(index + 1)]

    def _initialize_weights(self,gain:float):
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data, gain)
                m.bias.data.zero_()
                # n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                # m.weight.data.normal_(0,math.sqrt(2. /n))
                # if m.bias is not None:
                #     m.bias.data.zero_()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.decoder(encoded)
