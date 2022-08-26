import gin
from .data import DatasetXY
from .data_gen_np import get_liner_generator
from .data_gen_np import do_filter, do_filter_2, non_filter
from .data_gen_np import get_const_demographcs, get_test_demographcs, get_demographcs_from_ms_command

DatasetXY = gin.external_configurable(DatasetXY)
get_liner_generator = gin.external_configurable(get_liner_generator)

do_filter = gin.external_configurable(do_filter)
do_filter_2 = gin.external_configurable(do_filter_2)
get_const_demographcs = gin.external_configurable(get_const_demographcs)
get_test_demographcs = gin.external_configurable(get_test_demographcs)
get_demographcs_from_ms_command = gin.external_configurable(get_demographcs_from_ms_command)
