	j?t?:~@j?t?:~@!j?t?:~@	??0?ݖ???0?ݖ?!??0?ݖ?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$j?t?:~@333333??A??S??7~@Y???S㥻?*	     ?e@2F
Iterator::Model/?$???!?4?rO#H@);?O??n??1??=??D@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapX9??v???!??f??A@)X9??v???1??f??A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;?O???!?f??o*@)??~j?t??1?l???%@:Preprocessing2U
Iterator::Model::ParallelMapV2?~j?t???!a???@)?~j?t???1a???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zt?!??o??@){?G?zt?1??o??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!?@&?d@)????Mbp?1?@&?d@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??0?ݖ?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	333333??333333??!333333??      ??!       "      ??!       *      ??!       2	??S??7~@??S??7~@!??S??7~@:      ??!       B      ??!       J	???S㥻????S㥻?!???S㥻?R      ??!       Z	???S㥻????S㥻?!???S㥻?JCPU_ONLYY??0?ݖ?b 