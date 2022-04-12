import gin
from .data import DatasetXY
from .data_gen_np import get_liner_generator

DatasetXY = gin.external_configurable(DatasetXY)
get_liner_generator = gin.external_configurable(get_liner_generator)
