	j?t?Z?@j?t?Z?@!j?t?Z?@	?&?پ=???&?پ=??!?&?پ=??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$j?t?Z?@V-???Au?VX?@YˡE?????*	     p@2F
Iterator::ModelX9??v???!???H@)???Mb??1?I?I?IB@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapsh??|???!?%?%?%@@)sh??|???1?%?%?%@@:Preprocessing2U
Iterator::Model::ParallelMapV2???Q???!?X?X?X'@)???Q???1?X?X?X'@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty?&1???!6?5?5?%@)Zd;?O???1???!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?I+???!???!@)?I+???1???!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zt?!? ? ? ??){?G?zt?1? ? ? ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?&?پ=??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	V-???V-???!V-???      ??!       "      ??!       *      ??!       2	u?VX?@u?VX?@!u?VX?@:      ??!       B      ??!       J	ˡE?????ˡE?????!ˡE?????R      ??!       Z	ˡE?????ˡE?????!ˡE?????JCPU_ONLYY?&?پ=??b 