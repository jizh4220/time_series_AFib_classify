from typing import Callable

import tensorflow as tf
from pydantic import BaseModel, Field

KerasLayer = Callable[[tf.Tensor], tf.Tensor]


class MBConvParams(BaseModel):
    """MBConv parameters"""

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ex_ratio: float = Field(default=1, description="Expansion ratio")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    se_ratio: float = Field(default=8, description="Squeeze Excite ratio")
    droprate: float = Field(default=0, description="Drop rate")


class MBConvBlock(tf.keras.layers.Layer):
    def __init__(self, params: MBConvParams, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.conv_layers = []

        for _ in range(self.params.depth):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=self.params.filters,
                    kernel_size=self.params.kernel_size,
                    strides=self.params.strides,
                    padding='same'
                )
            )
            self.conv_layers.append(tf.keras.layers.BatchNormalization())
            self.conv_layers.append(tf.keras.layers.ReLU(max_value=6))  # ReLU6 activation

    def call(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        return x
