import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
from read_xml import read_xml

CLASSES = ('__background__','bad')


#CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, class_name, dets,ax,image_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

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
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    for bbox in read_xml(image_name+'.xml'):
        ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='blue', linewidth=3.5)
            )
    plt.axis('off')
    plt.tight_layout()
    plt.draw()



def vis_detections_cv(im, class_name, dets,image_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),3)
        #cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(bbox[0],bbox[1]-2),cv2.FONT_HERSHEY_COMPLEX,14,(255,0,0),2)
        cv2.putText(im,class_name+":"+str(score),(int(bbox[0]),int(bbox[1])-2),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
    return im




def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name+'.jpg')
    im_file='/home/makalo/workspace/code/kaggle/Faster-RCNN_TF/data/VOCdevkit2007/VOC2007/JPEGImages/'+image_name+'.jpg'
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    img = cv2.imread(im_file)
    im=np.copy(img)


    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    #im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im, cls, dets, ax,image_name, thresh=CONF_THRESH)
        im=vis_detections_cv(im,cls,dets,image_name,thresh=CONF_THRESH)
    for bbox in read_xml(image_name+'.xml'):
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),3)
    #comb=np.hstack([im,img])
    cv2.imshow('img',im)
    cv2.waitKey(0)



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # if args.model == ' ':
    #     raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, '/home/makalo/workspace/code/kaggle/Faster-RCNN_TF/output/faster_rcnn_end2end/voc_2007_trainval/VGGnet_fast_rcnn_iter_120000.ckpt')

    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']
    f_r=open('/home/makalo/workspace/code/kaggle/Faster-RCNN_TF/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt')
    line=f_r.readline()
    all_names=[]
    while(line):
        line=line.strip()
        all_names.append(line)
        line=f_r.readline()
    f_r.close()
    all_names=np.array(all_names)
    im_names = np.random.choice(all_names,10)
    #im_names=['37290d29-2a81-4c9d-aef6-15eea1376e0c','0b1d5b23-c3ab-41b5-b82d-a16a3d6b5674']
    print(im_names)



    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(sess, net, im_name)

    plt.show()

