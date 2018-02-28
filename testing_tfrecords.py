import tensorflow as tf
import config
import numpy as np
import data_provider
import losses
import models


data_folder = config.TFRECORDS_SAVE_PATH
frames, audio, ground_truth, ids = data_provider.get_split(data_folder, True,
                                                                 'train', 2,
						                  seq_length=2, debugging=True)








'''
path = config.TFRECORDS_SAVE_PATH + "/tf_records/test/16.tfrecords"
record_iterator = tf.python_io.tf_record_iterator(path=path)

for string_record in record_iterator:

  example = tf.train.Example()


  example.ParseFromString(string_record)

  sample_id = int(example.features.feature['sample_id']
                               .int64_list
                               .value[0])

  subject_id = int(example.features.feature['subject_id']
                              .int64_list
                              .value[0])

  label = (example.features.feature['label']
                                .bytes_list)

  audio = (example.features.feature['audio']
                              .float_list)

  print(sample_id)
'''