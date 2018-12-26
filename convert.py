import tensorflow as tf
import os
import json
from object_detection.utils import dataset_util
#from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = 2160 # Image height
  width = 3840 # Image width
  filename = example["ImageName"] # Filename of the image. Empty if image is not from file
  #image=Image.open("images/"+filename)
  with open("images/"+filename, "rb") as imageFile:
    f = imageFile.read()
  encoded_image_data = f # Encoded image bytes
  image_format = b'jpeg' 
  infolist=example["Info"]
  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
  crop=example["CropArea"].encode('utf-8')
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  for info in infolist:
    xmins.append(info["Bounds"][0])
    ymins.append(info["Bounds"][1])
    xmaxs.append(info["Bounds"][2])
    ymaxs.append(info["Bounds"][3])
    classes_text.append(info["Label"].encode('utf-8'))
    classes.append(1)
  print("Writing "+filename)
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/CropArea': dataset_util.bytes_feature(crop),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def main(_):
  fo=open("annotations/annotations.txt",'r')
  lines=fo.readlines()
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  # TODO(user): Write code to read in your dataset to examples variable
  for line in lines:
    example=json.loads(line)
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()