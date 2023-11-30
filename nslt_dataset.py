import json
import math
import os
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


# def load_rgb_frames(image_dir, vid, start, num):
#     frames = []
#     for i in range(start, start + num):
#         try:
#             img = cv2.imread(os.path.join(image_dir, vid, "image_" + str(i).zfill(5) + '.jpg'))[:, :, [2, 1, 0]]
#         except:
#             print(os.path.join(image_dir, vid, str(i).zfill(6) + '.jpg'))
#         w, h, c = img.shape
#         if w < 226 or h < 226:
#             d = 226. - min(w, h)
#             sc = 1 + d / min(w, h)
#             img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
#         img = (img / 255.) * 2 - 1
#         frames.append(img)
#     return np.asarray(frames, dtype=np.float32)


def load_rgb_frames_from_video(root, vid, resize=(256, 256)):
    video_path = os.path.join(root, vid)

    vidcap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for offset in range(min(64, int(total_frames - 0))):
        success, img = vidcap.read()
        if img is None:
            continue
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

        img = (img / 255.) * 2 - 1

        frames.append(img)

    return np.asarray(frames, dtype=np.float32)


# def load_flow_frames(image_dir, vid, start, num):
#     frames = []
#     for i in range(start, start + num):
#         imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
#         imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

#         w, h = imgx.shape
#         if w < 224 or h < 224:
#             d = 224. - min(w, h)
#             sc = 1 + d / min(w, h)
#             imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
#             imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

#         imgx = (imgx / 255.) * 2 - 1
#         imgy = (imgy / 255.) * 2 - 1
#         img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
#         frames.append(img)
#     return np.asarray(frames, dtype=np.float32)

def load_flow_frames_from_video(vid_root, vid, start, num, resize=(256, 256)):
        video_path = os.path.join(vid_root, vid)  
        
        # The video feed is read in as
        # a VideoCapture object
        cap = cv2.VideoCapture(video_path)
        # print(cap)
        
        # ret = a boolean return value from
        # getting the frame, first_frame = the
        # first frame in the entire video sequence
        ret, first_frame = cap.read()

        print(ret)
        
        # Converts frame to grayscale because we
        # only need the luminance channel for
        # detecting edges - less computationally 
        # expensive
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        # Creates an image filled with zero
        # intensities with the same dimensions 
        # as the frame
        mask = np.zeros_like(first_frame)
        
        # Sets image saturation to maximum
        mask[..., 1] = 255

        frames = []
        
        while(cap.isOpened() and num > 0):
            
            # ret = a boolean return value from getting
            # the frame, frame = the current frame being
            # projected in the video
            ret, frame = cap.read()
            
            # print(ret)
            
            if ret : 
            
                # Opens a new window and displays the input
                # frame
                # cv2.imshow("input", frame)
                
                # Converts each frame to grayscale - we previously 
                # only converted the first frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculates dense optical flow by Farneback method
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                                None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Computes the magnitude and angle of the 2D vectors
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Sets image hue according to the optical flow 
                # direction
                mask[..., 0] = angle * 180 / np.pi / 2
                
                # Sets image value according to the optical flow
                # magnitude (normalized)
                mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                
                # Converts HSV to RGB (BGR) color representation
                rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
                
                # Opens a new window and displays the output frame
                # cv.imshow("dense optical flow", rgb)
                
                # print("frame", rgb.shape)
                rgb = cv2.resize(rgb, (224,224), interpolation=cv2.INTER_AREA)
                
                frames.append(rgb)
                
                # Updates previous frame
                prev_gray = gray
                
                # Frames are read by intervals of 1 millisecond. The
                # programs breaks out of the while loop when the
                # # user presses the 'q' key
                # if cv.waitKey(1) & 0xFF == ord('q'):
                #     break
                
                num -= 1
                
            else :
                break
            
        # The following frees up resources and
        # # closes all windows
        # cap.release()
        # cv.destroyAllWindows()

        # print(len(frames))
        return np.asarray(frames, dtype=np.float32)


def get_num_class(split_file):
    classes = set()

    content = json.load(open(split_file))

    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)

    return len(classes)


class NSLT(data_utl.Dataset):

    def __init__(self, root, df, transforms=None):
        self.num_classes = 8

        self.data = df
        self.transforms = transforms
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        

        total_frames = 80

        # try:
        #     start_f = random.randint(0, nf - total_frames - 1) + start_frame
        # except ValueError:
        #     start_f = start_frame

        imgs = load_rgb_frames_from_video(self.root, self.data['vid'].iloc[index])
        label = self.data['label_num'].iloc[index]

        imgs, label = self.pad(imgs, label, total_frames)

        # imgs = self.transforms(imgs)

        # ret_lab = torch.from_numpy(np.array([label]))
        # ret_img = video_to_tensor(imgs)

        # return ret_img, ret_lab

        # flow = load_flow_frames_from_video(self.root, self.data['vid'].iloc[index], 0, total_frames)
        
        # temp = label.copy()

        imgs, label = self.pad(imgs, label, total_frames)
        # flow, _ = self.pad(flow, [0], total_frames)
        
        # flow = torch.from_numpy(flow)
        # print(flow.shape)
        # flow = flow.permute(3, 0, 1, 2)

        imgs = self.transforms(imgs)

        # ret_lab = torch.from_numpy(label)
        ret_img = video_to_tensor(imgs)

        return ret_img, label

    def __len__(self):
        return len(self.data)

    def pad(self, imgs, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                prob = np.random.random_sample()
                if prob > 0.5:
                    pad_img = imgs[0]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
                else:
                    pad_img = imgs[-1]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
        else:
            padded_imgs = imgs

        # label = label[:, 0]
        # label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_imgs, label

    @staticmethod
    def pad_wrap(imgs, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                pad = imgs[:min(num_padding, imgs.shape[0])]
                k = num_padding // imgs.shape[0]
                tail = num_padding % imgs.shape[0]

                pad2 = imgs[:tail]
                if k > 0:
                    pad1 = np.array(k * [pad])[0]

                    padded_imgs = np.concatenate([imgs, pad1, pad2], axis=0)
                else:
                    padded_imgs = np.concatenate([imgs, pad2], axis=0)
        else:
            padded_imgs = imgs

        label = label[:, 0]
        label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_imgs, label
    
    def load_flow_frames_from_video(vid_root, vid, start, num, resize=(256, 256)):
        video_path = os.path.join(vid_root, vid + '.mp4')  
        
        # The video feed is read in as
        # a VideoCapture object
        cap = cv2.VideoCapture(video_path)
        # print(cap)
        
        # ret = a boolean return value from
        # getting the frame, first_frame = the
        # first frame in the entire video sequence
        ret, first_frame = cap.read()

        # print(ret)
        
        # Converts frame to grayscale because we
        # only need the luminance channel for
        # detecting edges - less computationally 
        # expensive
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        # Creates an image filled with zero
        # intensities with the same dimensions 
        # as the frame
        mask = np.zeros_like(first_frame)
        
        # Sets image saturation to maximum
        mask[..., 1] = 255

        frames = []
        
        while(cap.isOpened() and num > 0):
            
            # ret = a boolean return value from getting
            # the frame, frame = the current frame being
            # projected in the video
            ret, frame = cap.read()
            
            # print(ret)
            
            if ret : 
            
                # Opens a new window and displays the input
                # frame
                # cv2.imshow("input", frame)
                
                # Converts each frame to grayscale - we previously 
                # only converted the first frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculates dense optical flow by Farneback method
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                                None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Computes the magnitude and angle of the 2D vectors
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Sets image hue according to the optical flow 
                # direction
                mask[..., 0] = angle * 180 / np.pi / 2
                
                # Sets image value according to the optical flow
                # magnitude (normalized)
                mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                
                # Converts HSV to RGB (BGR) color representation
                rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
                
                # Opens a new window and displays the output frame
                # cv.imshow("dense optical flow", rgb)
                
                # print("frame", rgb.shape)
                rgb = cv2.resize(rgb, (224,224), interpolation=cv2.INTER_AREA)
                
                frames.append(rgb)
                
                # Updates previous frame
                prev_gray = gray
                
                # Frames are read by intervals of 1 millisecond. The
                # programs breaks out of the while loop when the
                # # user presses the 'q' key
                # if cv.waitKey(1) & 0xFF == ord('q'):
                #     break
                
                num -= 1
                
            else :
                break
            
        # The following frees up resources and
        # # closes all windows
        # cap.release()
        # cv.destroyAllWindows()

        # print(len(frames))
        return np.asarray(frames, dtype=np.float32)
    


