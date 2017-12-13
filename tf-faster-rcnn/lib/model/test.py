# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  # seems to have height, width, and image scales
  # still not sure about the scale, maybe full image it is 1.
  blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])
  
  boxes = rois[:, 1:5] / im_scales[0]
  # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.05):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time))

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)


# def realtime_detection(sess, net, imdb, image_folder, weights_filename, max_per_image=100, thresh=0.05, visualization='True'):
#   np.random.seed(cfg.RNG_SEED)
#   """Test a Fast R-CNN network on an image database."""

#   # set the image length to be number of images in the folder
#   #imgfiles = [join(image_folder, f) for f in listdir(image_folder) if (isfile(join(image_folder, f)) and f[-3:]=='jpg')]
#   cap = cv2.VideoCapture("/home/vca_ann/CNN/rcnn2/tf-faster-rcnn/data/demo/768x576.avi")
#   # if visualization=='false':
#   #   os.makedirs(image_folder+'_detected')
#   #   savefiles = [join(image_folder+'_detected', f) for f in listdir(image_folder) if (isfile(join(image_folder, f)) and f[-3:]=='jpg')]
#   #num_images = len(imgfiles)
#   #num_images = 1

#   # all detections are collected into:
#   #  all_boxes[cls][image] = N x 5 array of detections in
#   #  (x1, y1, x2, y2, score)
#   #all_boxes = [[[] for _ in range(num_images)]
#   #       for _ in range(imdb.num_classes)]

#   # output_dir = get_output_dir(imdb, weights_filename)
#   # timers
#   _t = {'im_detect' : Timer(), 'misc' : Timer()}
#   #TODO: plot the image with detections
#     # Create figure and axes
#   fig,ax = plt.subplots(1)

#   #for i in range(num_images):
#   while(True):
#     #i = 0
#     #im = cv2.imread(imgfiles[i])
#     ret, im = cap.read()
#     _t['im_detect'].tic()
#     scores, boxes = im_detect(sess, net, im)
#     _t['im_detect'].toc()

#     _t['misc'].tic()

#     # skip j = 0, because it's the background class
#     for cls_ind, cls in enumerate(imdb._classes[1:]):
#       cls_ind += 1
#       cls_boxes = boxes[:, cls_ind*4:(cls_ind+1)*4]
#       #inds = np.where(scores[:, cls_ind] > thresh)[0]
#       cls_scores = scores[:, cls_ind]      
#       cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
#         .astype(np.float32)
#       keep = nms(cls_dets, 0.3)
#       cls_dets = cls_dets[keep, :]
#       #all_boxes[j][i] = cls_dets
#       #print("class_dets:%d", len(cls_dets))

#     # Limit to max_per_image detections *over all classes*
#     # if max_per_image > 0:
#     #   image_scores = np.hstack([all_boxes[j][i][:, -1]
#     #                 for j in range(1, imdb.num_classes)])
#     #   if len(image_scores) > max_per_image:
#     #     image_thresh = np.sort(image_scores)[-max_per_image]
#     #     for j in range(1, imdb.num_classes):
#     #       keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
#     #       all_boxes[j][i] = all_boxes[j][i][keep, :]
#     # _t['misc'].toc()

#     # print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
#     #     .format(i + 1, num_images, _t['im_detect'].average_time,
#     #         _t['misc'].average_time))
    
    
#     # Display the image
#       im_RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#       ax.imshow(im_RGB)

#     # # Create a Rectangle patches
#     # for j in range(1, imdb.num_classes):
#     #   for det in all_boxes[j][i]:
#     #     bbox = det[:4]
#     #     rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
#     #     ax.text(bbox[0], bbox[1], str(imdb._classes[j])+':'+str(det[4]), fontdict={'color':'red'})
#     #     # Add the patch to the Axes
#     #     ax.add_patch(rect)    

#     # if visualization == 'true':
#     #   #plt.show()      
#     #   plt.draw()
      
#     #   #raw_input("Press Enter to continue to the next image...")
#     # #else:
#     #   #plt.savefig(savefiles[i])
#     # plt.pause(1)
#     # plt.cla()
#     # plt.close()
#       thresh = 0.8
#       inds = np.where(cls_dets[:, -1] >= thresh)[0]
#       if len(inds) == 0:
#           return
#     #print("length of indices: %d \n",len(inds))

#       for i in inds:
#           bbox = cls_dets[i, :4]
#           score = cls_dets[i, -1]

#           ax.add_patch(
#               plt.Rectangle((bbox[0], bbox[1]),
#                             bbox[2] - bbox[0],
#                             bbox[3] - bbox[1], fill=False,
#                             edgecolor='red', linewidth=3.5)
#               )
#           ax.text(bbox[0], bbox[1], str(imdb._classes[i])+':'+str(score), fontdict={'color':'red'})

#       plt.pause(.1)
#       plt.draw()
#       plt.cla()


#   # det_file = os.path.join(output_dir, 'detections.pkl')
#   # with open(det_file, 'wb') as f:
#   #   pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

#   # print('Evaluating detections')
#   # imdb.evaluate_detections(all_boxes, output_dir)

def realtime_detection(sess, net, imdb, image_folder, weights_filename, max_per_image=100, thresh=0.05, visualization='True'):
   np.random.seed(cfg.RNG_SEED)
   #cap = cv2.VideoCapture("/home/vca_ann/CNN/rcnn2/tf-faster-rcnn/data/demo/768x576.avi")
   cap = cv2.VideoCapture("/home/vca_ann/dataset/parking_1920_part_2.mp4")

   fig, ax = plt.subplots()
   while(True):
      ret, im = cap.read()      
      #im = cv2.imread(im_file)

      # Detect all object classes and regress object bounds
      timer = Timer()
      timer.tic()
      scores, boxes = im_detect(sess, net, im)
      timer.toc()
      print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

      # Visualize detections for each class
      CONF_THRESH = 0.8
      NMS_THRESH = 0.3
      for cls_ind, cls in enumerate(imdb._classes[1:]):
         cls_ind += 1 # because we skipped background
         cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
         cls_scores = scores[:, cls_ind]
         dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
         keep = nms(dets, NMS_THRESH)
         dets = dets[keep, :]
         vis_detections(im, cls, dets,ax, thresh=CONF_THRESH)
      plt.cla()
    #video loop ends

def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    #ion()
    #either learn drawing images using img plot or use opencv imshow if not possible... prefer first option
    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots()#(figsize=(12, 12))
    #ax.imshow(None)
    ax.imshow(im)#(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
    #plt.axis('off')
    #plt.tight_layout()
    plt.pause(.1)
    plt.draw()

RESULT_FILE_NAME = '/home/wenxi/tensorflow_frcnn_detection_results.txt'
def realtime_car_detection(sess, net, imdb, image_folder, weights_filename, max_per_image=100, thresh=0.05, visualization='false', save_to_file=False):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""

  # set the image length to be number of images in the folder
  imgfiles = [join(image_folder, f) for f in listdir(image_folder) if (isfile(join(image_folder, f)) and f[-3:]=='jpg')]
  if visualization=='false' and not save_to_file:
    os.makedirs(image_folder+'_detected')
    savefiles = [join(image_folder+'_detected', f) for f in listdir(image_folder) if (isfile(join(image_folder, f)) and f[-3:]=='jpg')]
  num_images = len(imgfiles)

  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  # output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  if save_to_file:
    writef = open(RESULT_FILE_NAME, 'w')
    T = Timer()
    T.tic()
  for i in range(num_images):
    im = cv2.imread(imgfiles[i])

    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    # add all vihecle type into one array as car type
    car_type_idx = []
    for type_name in ['articulated_truck', 'bus', 'car',
                      'motorized_vehicle', 'pickup_truck',
                     'single_unit_truck', 'work_van']:
      car_type_idx.append(imdb._class_to_ind[type_name])

    car_scores = np.array([]).reshape((0, 1))
    car_boxes = np.array([]).reshape((0,4))
    for j in car_type_idx:
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      car_scores =  np.vstack((car_scores, cls_scores[:, np.newaxis]))
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      car_boxes = np.vstack((car_boxes, cls_boxes))
    # cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
      # .astype(np.float32, copy=False)
    car_dets = np.hstack((car_boxes, car_scores)).astype(np.float32, copy=False)
    keep = nms(car_dets, cfg.TEST.NMS)
    car_dets = car_dets[keep, :]
    all_boxes[j][i] = car_dets

    # # Limit to max_per_image detections *over all classes*
    # if max_per_image > 0:
    #   image_scores = np.hstack([all_boxes[j][i][:, -1]
    #                 for j in range(1, imdb.num_classes)])
    #   if len(image_scores) > max_per_image:
    #     image_thresh = np.sort(image_scores)[-max_per_image]
    #     for j in range(1, imdb.num_classes):
    #       keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
    #       all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].diff,
            _t['misc'].diff))

    #TODO: plot the image with detections
    # Create figure and axes
    if not save_to_file:
      fig,ax = plt.subplots(1)  
      
      # Display the image
      im_RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      ax.imshow(im_RGB)  

    # Create a Rectangle patches
    for j in range(1, imdb.num_classes):
      for det in all_boxes[j][i]:
        bbox = det[:4]
        if save_to_file:
          imgfile_name = imgfiles[i]
          extra_info = []
          info_L = imgfile_name.split('_')
          img_time = info_L[-1]
          img_time = img_time[:-4]
          img_time = img_time.replace('+', ':')
          info_L = info_L[-2].split('/')
          camid = info_L[-1]
          writef.write('{}, {:d}, {:d}, {:d}, {:d}, {:d}\n'.format(img_time, int(camid), int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
        else:
          rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
          ax.text(bbox[0], bbox[1], 'car :'+str(det[4]), fontdict={'color':'blue'})
          # Add the patch to the Axes
          ax.add_patch(rect)

    if not save_to_file:
      if visualization == 'true':
        plt.show()
        
        raw_input("Press Enter to continue to the next image...")
      else:
        plt.savefig(savefiles[i])
    
      plt.close()

  # close file after loop   
  T.toc()
  print('It takes {:.3f}s to detect {:d} images'.format(T.diff, num_images))
  if save_to_file:
    writef.close()
