	V-??A}@V-??A}@!V-??A}@	as??1???as??1???!as??1???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$V-??A}@bX9?ȶ?A???K7?}@Y?Zd;??*	     ?R@2F
Iterator::ModelX9??v???!??????D@)/?$???1     <@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????????!??????@@){?G?z??1??????:@:Preprocessing2U
Iterator::Model::ParallelMapV2{?G?z??!??????*@){?G?z??1??????*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?I+???!UUUUUU-@)y?&1?|?1??????"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?I+???!UUUUUUM@)?~j?t?x?1       @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zt?!??????@){?G?zt?1??????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mbp?!VUUUUU@)????Mbp?1VUUUUU@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??v????!VUUUUU1@)????Mb`?1VUUUUU@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9`s??1???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	bX9?ȶ?bX9?ȶ?!bX9?ȶ?      ??!       "      ??!       *      ??!       2	???K7?}@???K7?}@!???K7?}@:      ??!       B      ??!       J	?Zd;???Zd;??!?Zd;??R      ??!       Z	?Zd;???Zd;??!?Zd;??JCPU_ONLYY`s??1???b 