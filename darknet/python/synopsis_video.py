from ctypes import *
import math
import random
import time
import cv2
from PIL import Image
import csv
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm

home_path = '/home/tducnguyen/NguyenTran/Study/8_Yolo/darknet'
video_path = home_path + '/data/Pedestrian overpass - original video (sample) - BriefCam Syndex.mp4'
frame_path = home_path + '/data/frames'

def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL("/home/tducnguyen/NguyenTran/Study/8_Yolo/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def store_single_disk(dir, image, image_id, metadata):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (_, _, 3) to be stored
        image_id    integer unique ID for image
    """
    Image.fromarray(image).save(dir + '/{}.png'.format(image_id))

    # write metadata
    if metadata != None:
        with open(dir + '/{}.csv'.format(image_id), "wt") as csvfile:
            writer = csv.writer(
                csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            # writer.writerow([label])
            # write something

def read_single_disk(dir, image_id, meta_flag):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image
    """
    image = np.array(Image.open(dir + '/{}.png'.format(image_id)))
    metadata = []

    if meta_flag:
        with open(dir + '/{}.csv'.format(image_id), "r") as csvfile:
            reader = csv.reader(
                csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            metadata = int(next(reader)[0])

    return image, metadata


def mse(imagea, imageb):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imagea.astype("float") - imageb.astype("float")) ** 2)
    err /= float(imagea.shape[0] * imagea.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imagea, imageb):
    # compute the mean squared error and structural similarity
    # MSE and SSIM
    # index for the images
    m = mse(imagea, imageb)
    s = ssim(imagea, imageb)
    return m, s


if __name__ == "__main__":
    # 1. Load Yolo model -----------------------------------------------------------------------------------------------
    print("Load YoLo model")
    net = load_net(b"/home/tducnguyen/NguyenTran/Study/8_Yolo/darknet/cfg/yolov3-voc_1class.cfg",
                   b"/home/tducnguyen/NguyenTran/Study/8_Yolo/darknet/yolov3-voc_1class.backup", 0)
    meta = load_meta(b"/home/tducnguyen/NguyenTran/Study/8_Yolo/darknet/cfg/voc.data")

    # r = detect(net, meta, b"/home/tducnguyen/NguyenTran/Study/8_Yolo/darknet/data/vlcsnap-2019-09-27-12h29m30s924.png")
    # print(r)

    # original = cv2.imread("/home/tducnguyen/NguyenTran/Study/8_Yolo/darknet/data/vlcsnap-2019-09-27-12h29m30s924.png")

    # 2. Read frame from input video -----------------------------------------------------------------------------------
    num_frame = 0
    if len(os.listdir(frame_path)) == 0:
        print("No frame stored")
        print("Read video frame......")
        Path(frame_path).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        while True:
            _, frame = cap.read()
            if frame is None:
                break

            # Store image to disk:
            # cv2.imshow('app', frame)
            store_single_disk(frame_path, frame, frame_id, None)
            frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        num_frame = len(os.listdir(frame_path))
        print("Folder content {} frames".format(num_frame))

    # 3. Check and remove the background frames ("Static Background Removal from a video")------------------------------
    checked_fr = []
    pre_frame = []
    same_cont = 0
    for fr_id in tqdm(range(num_frame)):
        if fr_id == 0:
            pre_frame = np.array(read_single_disk(frame_path, fr_id, False)[0])
            continue

        if fr_id >= 100:
            break

        new_frame = np.array(read_single_disk(frame_path, fr_id, False)[0])

        # convert to gray scale before compare
        pre_frame_grey = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
        new_frame_grey = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Check if pre frame and current frame is same or not
        mse_, ssim_ = compare_images(pre_frame_grey, new_frame_grey)

        if mse_ >= 2 or ssim_ <= 0.98:
            checked_fr.append(fr_id)

        # Update previous frame after finish compare
        pre_frame = new_frame

    print('Remove {} from {} frames'.format(num_frame - len(checked_fr), num_frame))

    # 4. Load frames, do the object recognition and store the corresponding metadata------------------------------------
    detected_fr = []
    detected_metadata = []
    for i in tqdm(range(len(checked_fr))):
        fr_id = checked_fr[i]
        r = detect(net, meta,
                   "/home/tducnguyen/NguyenTran/Study/8_Yolo/darknet/data/frames/{}.png".format(fr_id).encode())

        # Store to metadata if object recognizable
        if r:
            detected_fr.append(fr_id)
            detected_metadata.append(r)

    # Assign the id to each object (consider the distance between object in 2 adjacent frame
    pre_num_obj = 0
    pre_object_identity = []
    object_identity = []
    object_tracking = []

    for i in tqdm(range(len(detected_fr))):
        num_obj = len(detected_metadata[i])
        if i == 0:
            for ob_id in range(num_obj):
                pre_object_identity.append([detected_metadata[i][ob_id][2]])
            continue
        else:
            matched_object = []
            for object_ in range(num_obj):
                # Calculate the distance of each object to all object in previous frame
                obj_coord_chk = detected_metadata[i][object_][2]
                not_detected = 0
                for obj_coor_ref_id, obj_coor_ref in enumerate(pre_object_identity):
                    distance = np.linalg.norm(np.array(obj_coord_chk) - np.array(obj_coor_ref[-1]))
                    if distance <= 80:  # the same object
                        pre_object_identity[obj_coor_ref_id].append(obj_coord_chk)
                        matched_object.append(object_)

                    else:
                        not_detected += 1
                    if not_detected >= len(pre_object_identity):
                        # Add new object oder to object_tracking list
                        pre_object_identity.append([obj_coord_chk])

            # Check if any object can not be detected in new frame, that object has been finish,
            # will be add to tracking list
            for object_id, _ in enumerate(pre_object_identity):
                if object_id not in matched_object:
                    # add trace to tracking list and remove trace
                    object_tracking.append(pre_object_identity[object_id])

                    # Remove the completed trace
                    pre_object_identity[object_id] = []

                    # Rearrange the pre_object_identity
                    if object_id >= len(pre_object_identity) - 1:
                        tmp_pre_object_identity = []
                        for tmp_id in range(len(pre_object_identity)):
                            if pre_object_identity[tmp_id] != []:
                                tmp_pre_object_identity.append(pre_object_identity[tmp_id])


    print('Finish')




