import os
from os.path import join
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import torch
import time
from data_utils import gtransforms
from data_utils.data_parser import WebmDataset
from numpy.random import choice as ch
import random
import json


class VideoFolder(torch.utils.data.Dataset):
    """
    Something-Something dataset based on *frames* extraction
    """
    
    bbox_folder_path = '/opt/data/private/something_else/mycode/bounding_box_smthsmth'
    
    def __init__(self,
                 root,
                 file_input,
                 file_labels,
                 frames_duration,
                 args=None,
                 multi_crop_test=False,
                 sample_rate=2,
                 is_test=False,
                 is_val=False,
                 num_boxes=10,
                 model=None,
                 if_augment=True,
                 if_crop_paste_per_video=False,
                 if_crop_paste_per_image=False,
                 if_object_mixup=False):
        """
        :param root: data root path
        :param file_input: inputs path
        :param file_labels: labels path
        :param frames_duration: number of frames
        :param multi_crop_test:
        :param sample_rate: FPS
        :param is_test: is_test flag
        :param k_split: number of splits of clips from the video
        :param sample_split: how many frames sub-sample from each clip
        :param sample_split: how many frames sub-sample from each clip
        :param sample_split: how many frames sub-sample from each clip
        """
        self.in_duration = frames_duration
        # self.coord_nr_frames = self.in_duration // 2
        self.coord_nr_frames = self.in_duration
        # self.coord_nr_frames = self.in_duration * 2
        self.multi_crop_test = multi_crop_test
        self.sample_rate = sample_rate
        self.if_augment = if_augment
        self.is_val = is_val
        self.data_root = root
        self.dataset_object = WebmDataset(file_input, file_labels, root, is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.model = model
        self.num_boxes = num_boxes

        self.if_object_mixup = if_object_mixup
        if self.if_object_mixup:
            self.alpha = 0.2
            self.mixup_weight = 0.5

        self.if_crop_paste_per_video = if_crop_paste_per_video
        if self.if_crop_paste_per_video or self.if_object_mixup:
            self.cropped_img_list_length = 8
            self.read_index = 0
            self.write_index = 0
            self.cropped_img_list = [Image.new("RGB", (128, 128)) for i in range(self.cropped_img_list_length)]

        self.if_crop_paste_per_image = if_crop_paste_per_image
        if self.if_crop_paste_per_image:
            self.cropped_img_list_length = 32
            self.read_index = 0
            self.write_index = 0
            self.cropped_img_list = [Image.new("RGB", (128, 128)) for i in range(self.cropped_img_list_length)]

        # Prepare data for the data loader
        self.args = args
        self.prepare_data()
        self.pre_resize_shape = (256, 340)

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        # Transformations
        if not self.is_val:
            self.transforms = [
                gtransforms.GroupResize((224, 224)),
            ]
        elif self.multi_crop_test:
            self.transforms = [
                gtransforms.GroupResize((256, 256)),
                gtransforms.GroupRandomCrop((256, 256)),
            ]
        else:
            self.transforms = [
                gtransforms.GroupResize((224, 224))
                # gtransforms.GroupCenterCrop(256),
            ]
        self.transforms += [
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(self.img_mean, self.img_std),
        ]
        self.transforms = Compose(self.transforms)

        if self.if_augment:
            if not self.is_val:  # train, multi scale cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=224,
                                                                   scales=[1, .875, .75])
            else:  # val, only center cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=224,
                                                                   scales=[1],
                                                                   max_distort=0,
                                                                   center_crop_only=True)
        else:
            self.random_crop = None

    def load_one_video_json(self, folder_id):
        with open(os.path.join(VideoFolder.bbox_folder_path, folder_id+'.json'), 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        return video_data


    def prepare_data(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        This process will take up a long time, I want to save these objects into a file and read it
        :return:
        """
        print("Loading label strings")
        self.label_strs = ['_'.join(class_name.split(' ')) for class_name in self.classes]  # will this attribute be used?
        vid_names = []
        labels = []
        frame_cnts = []

        if self.args.is_sub:
            prefix = 'sub_'
        else:
            prefix = ''
        if self.args.meta_tr:
            fs_prefix = 'fewshot'
        else:
            fs_prefix = '/'
        if not self.is_val:
            with open('/opt/data/private/something_else/mycode/misc/{}/{}train_vid_name.json'.format(fs_prefix, prefix), 'r') as f:
                vid_names = json.load(f)
            with open('/opt/data/private/something_else/mycode/misc/{}/{}train_labels.json'.format(fs_prefix, prefix), 'r') as f:
                labels = json.load(f)
            with open('/opt/data/private/something_else/mycode/misc/{}/{}train_frame_cnts.json'.format(fs_prefix, prefix), 'r') as f:
                frame_cnts = json.load(f)
        else:
            with open('/opt/data/private/something_else/mycode/misc/{}/val_vid_name.json'.format(fs_prefix), 'r') as f:
                vid_names = json.load(f)
            with open('/opt/data/private/something_else/mycode/misc/{}/val_labels.json'.format(fs_prefix), 'r') as f:
                labels = json.load(f)
            with open('/opt/data/private/something_else/mycode/misc/{}/val_frame_cnts.json'.format(fs_prefix), 'r') as f:
                frame_cnts = json.load(f)
        self.vid_names = vid_names
        self.labels = labels
        self.frame_cnts = frame_cnts

    # todo: might consider to replace it to opencv, should be much faster
    def load_frame(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        return Image.open(join(os.path.dirname(self.data_root), 'frames', vid_name, '%04d.jpg' % (frame_idx + 1))).convert('RGB')


    def load_object_mixup_frame(self, vid_name, frame_idx, video_data):

        img = Image.open(join(os.path.dirname(self.data_root), 'frames',
                              vid_name, '%04d.jpg' % (frame_idx + 1))).convert('RGB')
        img_recover = img.copy()
        img_copy = img.copy()
        pic_id = os.path.join(vid_name, '%04d.jpg' % (frame_idx + 1))
        label_list = []
        for pic_data in video_data:
            if pic_data['name'] == pic_id:
                label_list = pic_data['labels']
                break
        
        for i, label in enumerate(label_list):
            if not label['category'] == 'hand':
                x1 = int(label['box2d']['x1'])
                y1 = int(label['box2d']['y1'])
                x2 = int(label['box2d']['x2']) + 1
                y2 = int(label['box2d']['y2']) + 1


                img.paste(Image.blend(
                    img_copy.crop((x1, y1, x2, y2)),
                    self.cropped_img_list[(self.read_index + i) % self.cropped_img_list_length].resize((x2 - x1, y2 - y1)),
                    self.mixup_weight), (x1, y1, x2, y2))

        for label in label_list:
            x1 = int(label['box2d']['x1'])
            y1 = int(label['box2d']['y1'])
            x2 = int(label['box2d']['x2'])
            y2 = int(label['box2d']['y2'])
            if label['category'] == 'hand':
                img_cropped = img_copy.crop((x1, y1, x2, y2))
                img.paste(img_cropped, (x1, y1, x2, y2))
                img_copy = img_recover.copy()

        img.save('/opt/data/private/test/img_{}_{}.jpg'.format(vid_name, str(frame_idx)))

        return img


    def load_pasted_frame_per_video(self, vid_name, frame_idx, video_data):

        img = Image.open(join(os.path.dirname(self.data_root), 'frames',
                              vid_name, '%04d.jpg' % (frame_idx + 1))).convert('RGB')
        img_recover = img.copy()
        img_copy = img.copy()
        pic_id = os.path.join(vid_name, '%04d.jpg' % (frame_idx + 1))
        label_list = []
        for pic_data in video_data:
            if pic_data['name'] == pic_id:
                label_list = pic_data['labels']
                break
        # paste the instances onto the img
        for i, label in enumerate(label_list):
            if not label['category'] == 'hand':
                x1 = int(label['box2d']['x1'])
                y1 = int(label['box2d']['y1'])
                x2 = int(label['box2d']['x2']) + 1
                y2 = int(label['box2d']['y2']) + 1

                img.paste(self.cropped_img_list[(self.read_index+i) % self.cropped_img_list_length].resize((x2-x1, y2-y1)), (x1, y1, x2, y2))

        # the information from hand shouldn't be discarded
        for label in label_list:
            x1 = int(label['box2d']['x1'])
            y1 = int(label['box2d']['y1'])
            x2 = int(label['box2d']['x2'])
            y2 = int(label['box2d']['y2'])
            if label['category'] == 'hand':
                img_cropped = img_copy.crop((x1, y1, x2, y2))
                img.paste(img_cropped, (x1, y1, x2, y2))
                img_copy = img_recover.copy()

        return img

    def load_pasted_frame_per_image(self, vid_name, frame_idx, video_data):

        img = Image.open(join(os.path.dirname(self.data_root), 'frames',
                              vid_name, '%04d.jpg' % (frame_idx + 1))).convert('RGB')
        img_recover = img.copy()
        img_copy = img.copy()
        pic_id = os.path.join(vid_name, '%04d.jpg' % (frame_idx + 1))
        label_list = []
        for pic_data in video_data:
            if pic_data['name'] == pic_id:
                label_list = pic_data['labels']
                break

        for label in label_list:
            if not label['category'] == 'hand':
                x1 = int(label['box2d']['x1'])
                y1 = int(label['box2d']['y1'])
                x2 = int(label['box2d']['x2']) + 1
                y2 = int(label['box2d']['y2']) + 1
                self.read_index = random.randint(0, self.cropped_img_list_length - 1)
                img.paste(self.cropped_img_list[self.read_index].resize((x2-x1, y2-y1)), (x1, y1, x2, y2))

        for label in label_list:
            x1 = int(label['box2d']['x1'])
            y1 = int(label['box2d']['y1'])
            x2 = int(label['box2d']['x2'])
            y2 = int(label['box2d']['y2'])
            if label['category'] == 'hand':
                img_cropped = img_copy.crop((x1, y1, x2, y2))
                img.paste(img_cropped, (x1, y1, x2, y2))
                img_copy = img_recover.copy()

        # img.save('/opt/data/private/test1/img_{}_{}.jpg'.format(vid_name, str(frame_idx)))
        return img


    def _update_cropped_list(self, vid_name, frame_idx, video_data):

        label_list = []
        self.write_index = (self.write_index + 1) % self.cropped_img_list_length
        self.read_index = random.randint(0, self.cropped_img_list_length - 1)
        if self.if_object_mixup:
            # self.mixup_weight = np.random.beta(self.alpha, self.alpha)
            self.mixup_weight = 0.5
        
        # crop hands and objects
        img = Image.open(join(os.path.dirname(self.data_root), 'frames', vid_name,
                               '%04d.jpg' % (frame_idx + 1))).convert('RGB')
        img_recover = img.copy()
        pic_id = os.path.join(vid_name, '%04d.jpg' % (frame_idx + 1))
        for pic_data in video_data:
            if pic_data['name'] == pic_id:
                label_list = pic_data['labels']
                break

        for label in label_list:
            x1 = int(label['box2d']['x1'])
            y1 = int(label['box2d']['y1'])
            x2 = int(label['box2d']['x2'])
            y2 = int(label['box2d']['y2'])
            if not label['category'] == 'hand':
                self.cropped_img_list[self.write_index] = img.crop((x1, y1, x2, y2))
                img = img_recover.copy()
                self.write_index = (self.write_index + 1) % self.cropped_img_list_length


    def _sample_indices(self, nr_video_frames):
        average_duration = nr_video_frames * 1.0 / self.coord_nr_frames
        if average_duration > 0:
            offsets = np.multiply(list(range(self.coord_nr_frames)), average_duration) \
                      + np.random.uniform(0, average_duration, size=self.coord_nr_frames)
            offsets = np.floor(offsets)
        elif nr_video_frames > self.coord_nr_frames:
            offsets = np.sort(np.random.randint(nr_video_frames, size=self.coord_nr_frames))
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def _get_val_indices(self, nr_video_frames):
        if nr_video_frames > self.coord_nr_frames:
            tick = nr_video_frames * 1.0 / self.coord_nr_frames
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.coord_nr_frames)])
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def sample_single(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """
        n_frame = self.frame_cnts[index] - 1
        d = self.in_duration * self.sample_rate  # 16 * 2, sample_rate is the step size, and d is the sample interval, in_duration is the sample length
        if n_frame > d:
            if not self.is_val:
                # random sample
                offset = np.random.randint(0, n_frame - d)
            else:
                # center crop
                offset = (n_frame - d) // 2
            frame_list = list(range(offset, offset + d, self.sample_rate))
        else:
            # Temporal Augmentation
            if not self.is_val: # train
                if n_frame - 2 < self.in_duration:  # why n_frame-2???
                    # less frames than needed
                    pos = np.linspace(0, n_frame - 2, self.in_duration)
                else: # take one
                    pos = np.sort(np.random.choice(list(range(n_frame - 2)), self.in_duration, replace=False))
            else:
                pos = np.linspace(0, n_frame - 2, self.in_duration)
            frame_list = [round(p) for p in pos]

        frame_list = [int(x) for x in frame_list]

        if not self.is_val:  # train
            coord_frame_list = self._sample_indices(n_frame)
        else:  # val
            coord_frame_list = self._get_val_indices(n_frame)

        # assert len(coord_frame_list) == len(frame_list) // 2
        assert len(coord_frame_list) == len(frame_list)

        folder_id = str(int(self.vid_names[index]))  # lxs1
        
        # video_data = self.box_annotations[folder_id]
        while True:
            try:
                video_data = self.load_one_video_json(folder_id)
                break
            except OSError:
                time.sleep(0.1)

        # union the objects of two frames
        object_set = set()
        for frame_id in coord_frame_list:
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']  # standard category: [0001, 0002]
                object_set.add(standard_category)
        object_set = sorted(list(object_set))

        # NOTE: IMPORTANT!!! To accelerate the data loader, here it only reads one image
        #  (they're of the same scale in a video) to get its height and width
        #  It must be modified for models using any appearance features.
        frames = []
        for fidx in coord_frame_list:
            frames.append(self.load_frame(self.vid_names[index], fidx))
            break  # only one image
        height, width = frames[0].height, frames[0].width

        frames = [img.resize((self.pre_resize_shape[1], self.pre_resize_shape[0]), Image.BILINEAR) for img in frames]  # just one frame in List:frames

        if self.random_crop is not None:
            frames, (offset_h, offset_w, crop_h, crop_w) = self.random_crop(frames)
        else:
            offset_h, offset_w, (crop_h, crop_w) = 0, 0, self.pre_resize_shape
        if self.model not in ['coord', 'coord_latent', 'coord_latent_nl', 'coord_latent_concat']:
            frames = []
            elif self.if_crop_paste_per_video:
                for fidx in frame_list:
                    while True:
                        try:
                            frames.append(self.load_pasted_frame_per_video(self.vid_names[index], fidx, video_data))
                            break
                        except OSError:
                            time.sleep(0.1)
                
                while True:
                    try:
                        self._update_cropped_list(self.vid_names[index], frame_list[0], video_data)
                        break
                    except OSError:
                        time.sleep(0.1)

            elif self.if_crop_paste_per_image:
                for fidx in frame_list:
                    while True:
                        try:
                            frames.append(self.load_pasted_frame_per_image(self.vid_names[index], fidx, video_data))
                            break
                        except OSError:
                            print('OSError')
                            time.sleep(0.1)
                
                while True:
                    try:
                        self._update_cropped_list(self.vid_names[index], frame_list[0], video_data)
                        break
                    except OSError:
                        time.sleep(0.1)
            elif self.if_object_mixup:
                for fidx in frame_list:
                    while True:
                        try:
                            frames.append(self.load_object_mixup_frame(self.vid_names[index], fidx, video_data))
                            break
                        except OSError:
                            time.sleep(0.1)
                
                while True:
                    try:
                        self._update_cropped_list(self.vid_names[index], frame_list[0], video_data)
                        break
                    except OSError:
                        time.sleep(0.1)

            else:
                for fidx in frame_list:
                    while True:
                        try:
                            frames.append(self.load_frame(self.vid_names[index], fidx))
                            break
                        except OSError:
                            time.sleep(0.1)


        else:
            # Now for accelerating just pretend we have had frames
            frames = frames * self.in_duration  # TODO:repeat the first loaded frame nr_frames times

        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(width), self.pre_resize_shape[0] / float(height)
        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h)

        box_tensors = torch.zeros((self.coord_nr_frames, self.num_boxes, 4), dtype=torch.float32)  # (cx, cy, w, h)
        box_categories = torch.zeros((self.coord_nr_frames, self.num_boxes))
        for frame_index, frame_id in enumerate(coord_frame_list):
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                global_box_id = object_set.index(standard_category)

                box_coord = box_data['box2d']
                x0, y0, x1, y1 = box_coord['x1'], box_coord['y1'], box_coord['x2'], box_coord['y2']

                # scaling due to initial resize
                x0, x1 = x0 * scale_resize_w, x1 * scale_resize_w
                y0, y1 = y0 * scale_resize_h, y1 * scale_resize_h

                # shift
                x0, x1 = x0 - offset_w, x1 - offset_w
                y0, y1 = y0 - offset_h, y1 - offset_h

                x0, x1 = np.clip([x0, x1], a_min=0, a_max=crop_w-1)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=crop_h-1)

                # scaling due to crop
                x0, x1 = x0 * scale_crop_w, x1 * scale_crop_w
                y0, y1 = y0 * scale_crop_h, y1 * scale_crop_h

                # precaution
                x0, x1 = np.clip([x0, x1], a_min=0, a_max=223)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=223)

                # (cx, cy, w, h)
                gt_box = np.array([(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0], dtype=np.float32)

                # normalize gt_box into [0, 1]
                gt_box /= 224

                # load box into tensor
                try:
                    box_tensors[frame_index, global_box_id] = torch.tensor(gt_box).float()
                    box_categories[frame_index, global_box_id] = 1 if box_data['standard_category'] == 'hand' else 2
                except IndexError:
                    pass
                # load box category
                # try:
                #    box_categories[frame_index, global_box_id] = 1 if box_data['standard_category'] == 'hand' else 2  # 0 is for none, each frame has 10 bounding boxes, seems the max number of instance is 7
                # except:
                #    pass

                # load image into tensor
                x0, y0, x1, y1 = list(map(int, [x0, y0, x1, y1]))  # region of interest? # no use...
        return frames, box_tensors, box_categories


    def __getitem__(self, index):
        '''
        box_tensors: [nr_frames, num_boxes, 4]
        box_categories: [nr_frames, num_boxes], value is 0(none), 1 (hand), 2 (object)
        frames: what about the frames shape?
        '''
        frames, box_tensors, box_categories = self.sample_single(index)
        frames = self.transforms(frames)  # original size is (t, c, h, w)
        global_img_tensors = frames.permute(1, 0, 2, 3)   # (c, t, h, w)
        return global_img_tensors, box_tensors, box_categories, self.classes_dict[self.labels[index]]

    def __len__(self):
        return len(self.json_data)

    def unnormalize(self, img, divisor=255):
        """
        The inverse operation of normalization
        Both the input & the output are in the format of BxCxHxW
        """
        for c in range(len(self.img_mean)):
            img[:, c, :, :].mul_(self.img_std[c]).add_(self.img_mean[c])

        return img / divisor

    def img2np(self, img):
        """
        Convert image in torch tensors of BxCxTxHxW [float32] to a numpy array of BxHxWxC [0-255, uint8]
        Take the first frame along temporal dimension
        if C == 1, that dimension is removed
        """
        img = self.unnormalize(img[:, :, 0, :, :], divisor=1).to(torch.uint8).permute(0, 2, 3, 1)
        if img.shape[3] == 1:
            img = img.squeeze(3)
        return img.cpu().numpy()