import gin
from .data import DatasetXY, DummyDataset
from .data_gen_np import get_liner_generator, limits
from .data_gen_np import do_filter, do_filter_2, non_filter
from .data_gen_np import get_humanlike, get_humanlike_2, get_humanlike_2_1, get_const_demographcs, get_test_demographcs, get_demographcs_from_ms_command, get_test_demographcs_2, get_random_events_by_points, get_one_of_two, get_strong_bootleneck, get_one_of_many, get_set1, get_set2, get_set2_plus, get_set3, get_zigzag

DatasetXY = gin.external_configurable(DatasetXY)
DummyDataset = gin.external_configurable(DummyDataset)
get_liner_generator = gin.external_configurable(get_liner_generator)

do_filter = gin.external_configurable(do_filter)
do_filter_2 = gin.external_configurable(do_filter_2)
non_filter = gin.external_configurable(non_filter)
get_const_demographcs = gin.external_configurable(get_const_demographcs)
get_test_demographcs = gin.external_configurable(get_test_demographcs)
get_test_demographcs_2 = gin.external_configurable(get_test_demographcs_2)
get_demographcs_from_ms_command = gin.external_configurable(get_demographcs_from_ms_command)
get_humanlike = gin.external_configurable(get_humanlike)
get_humanlike_2 = gin.external_configurable(get_humanlike_2)
get_humanlike_2_1 = gin.external_configurable(get_humanlike_2_1)
get_random_events_by_points = gin.external_configurable(get_random_events_by_points)
get_one_of_two = gin.external_configurable(get_one_of_two)
get_strong_bootleneck = gin.external_configurable(get_strong_bootleneck)
get_one_of_many = gin.external_configurable(get_one_of_many)

get_set1 = gin.external_configurable(get_set1)
get_set2 = gin.external_configurable(get_set2)
get_set2_plus = gin.external_configurable(get_set2_plus)
get_set3 = gin.external_configurable(get_set3)
get_zigzag = gin.external_configurable(get_zigzag)