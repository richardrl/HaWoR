import cv2
from tqdm import tqdm
import numpy as np
import torch

from ultralytics import YOLO


if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def hawor_json_to_np_arrays(hawor_json):
    boxes = []
    confs = []
    handedness = []
    track_id = []
    for found_hand in range(len(hawor_json)):
        boxes.append(np.array(hawor_json[found_hand][0]))
        confs.append(np.array(hawor_json[found_hand][2]))
        handedness.append(np.array(1 if hawor_json[found_hand][1] == "right_hand" else 0))

        # TODO: maybe right hand needs to be 0 and left hand needs to be 1
        track_id.append(handedness[-1])
    boxes = np.stack(boxes)
    confs = np.stack(confs)
    handedness = np.stack(handedness)
    track_id = np.stack(track_id)
    return boxes, confs, handedness, track_id


import json
def detect_track(imgfiles, thresh=0.5, detector='yolo'):
    if detector == 'yolo':
        hand_det_model = YOLO('./weights/external/detector.pt')
    elif detector == 'hands23':
        # just use json
        # TODO: remove hardcoded path
        # bbox_json = json.load(open('/data/scratch-oc40/pulkitag/rli14/hamer_diffusion_policy/labels/10312024_sfu_cooking_test/bbox.json'))

        bbox_json = json.load(open('/data/scratch-oc40/pulkitag/rli14/hamer_diffusion_policy/labels/minnesota_cooking_074_2/bbox.json'))

    # Run
    boxes_ = []
    tracks = {}
    for t, imgpath in enumerate(tqdm(imgfiles)):
        img_cv2 = cv2.imread(imgpath)

        ### --- Detection ---
        with torch.no_grad():
            with autocast():
                if detector == 'yolo':
                    results = hand_det_model.track(img_cv2, conf=thresh, persist=True, verbose=False)
                    # -> 1, 4
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    # -> 1
                    confs = results[0].boxes.conf.cpu().numpy()
                    # -> 1
                    handedness = results[0].boxes.cls.cpu().numpy()

                    # if boxes.shape[0] == 0:
                    #     import pdb
                    #     pdb.set_trace()
                    if not results[0].boxes.id is None:
                        track_id = results[0].boxes.id.cpu().numpy()
                    else:
                        track_id = [-1] * len(boxes)
                else:
                    current_bbox_json = bbox_json[f"{t+1:06d}"]
                    if not current_bbox_json:
                        boxes = np.zeros((0, 4))
                        confs = np.zeros(0)
                        handedness = np.zeros(0)
                        track_id = [-1] * len(boxes)
                    else:
                        boxes, confs, handedness, track_id = hawor_json_to_np_arrays(current_bbox_json)
                        print(f"track_id: {track_id}")
                boxes = np.hstack([boxes, confs[:, None]])
                find_right = False
                find_left = False
                for idx, box in enumerate(boxes):
                    if track_id[idx] == -1:
                        if handedness[[idx]] > 0:
                            id = int(10000)
                        else:
                            id = int(5000)
                    else:
                        id = track_id[idx]
                    subj = dict()
                    subj['frame'] = t 
                    subj['det'] = True
                    subj['det_box'] = boxes[[idx]]
                    subj['det_handedness'] = handedness[[idx]]
                    
                    
                    if (not find_right and handedness[[idx]] > 0) or (not find_left and handedness[[idx]]==0):
                        if id in tracks:
                            tracks[id].append(subj)
                        else:
                            tracks[id] = [subj]

                        if handedness[[idx]] > 0:
                            find_right = True
                        elif handedness[[idx]] == 0:
                            find_left = True
    tracks = np.array(tracks, dtype=object)
    boxes_ = np.array(boxes_, dtype=object)

    return boxes_, tracks


def parse_chunks(frame, boxes, min_len=16):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []
    boxes_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        b_chunk = boxes[start:bk]
        start = bk
        if len(f_chunk)>=min_len:
            frame_chunks.append(f_chunk)
            boxes_chunks.append(b_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            b_chunk = boxes[bk:]
            if len(f_chunk)>=min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)

    return frame_chunks, boxes_chunks

def parse_chunks_hand_frame(frame):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []

    # by getting the differences between successive frames, we calculate the number of steps per chunk
    # most of the time, step will be 1, because we have contiguous frames
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])

    # when the step is not 1, we have multiple missing frames
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        # note: the breaks are local indices in the frame buffer
        f_chunk = frame[start:bk]
        start = bk
        if len(f_chunk) > 0:
            frame_chunks.append(f_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            if len(f_chunk) > 0:
                frame_chunks.append(f_chunk)

    # TLDR:
    # this function should cover the ENTIRE sequence, by combining single step frames into contiguous chunks
    # if a frame is by itself, it is its own "chunk"
    return frame_chunks
