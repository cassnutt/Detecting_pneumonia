	q=
ף`}@q=
ף`}@!q=
ף`}@	1??????1??????!1??????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q=
ף`}@j?t???A?t?^}@Y+??????*	     ?R@2F
Iterator::Model??~j?t??!Ϻ???I@)y?&1???1?n0E>?B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??~j?t??!Ϻ???9@)???Q???1S??n0E4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?~j?t???!v?)?Y70@)?~j?t???1v?)?Y70@:Preprocessing2U
Iterator::Model::ParallelMapV2{?G?z??!o0E>?+@){?G?z??1o0E>?+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zt?!o0E>?@){?G?zt?1o0E>?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!?Y7?"?@)????Mbp?1?Y7?"?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no91??????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	j?t???j?t???!j?t???      ??!       "      ??!       *      ??!       2	?t?^}@?t?^}@!?t?^}@:      ??!       B      ??!       J	+??????+??????!+??????R      ??!       Z	+??????+??????!+??????JCPU_ONLYY1??????b 