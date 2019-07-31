import cv2
import numpy as np
import os

from collections import namedtuple

from torch.utils.data import Dataset
from pysot.core.config import cfg

from pysot.datasets.otb import OTBDataset
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.utils.bbox import center2corner, Center

DEFAULT_FRAME_RANGE = 10
AUGMENTATION_IMAGE_SUFFIX = "_aug"
BBOX = namedtuple('Bbox', 'x1 y1 x2 y2')


class TrackingDatasetAdapter:
    r"""
    Class to adapt videos dataset to the tracking dataset
    """
    def __init__(self, dataset, frame_root):
        videos = dataset.videos
        # for video_name in list(videos.keys()):
        #     video_data = videos[video_name]
        #     frames = video_data.get_frames()

        # self.labels = meta_data

        # Number of videos in the dataset
        self.num = len(videos.keys())
        self.frame_root = frame_root

        # List of video names in the dataset
        self.video_names = list(videos.keys())

        # Mapping of video names to video objects
        self.videos = videos

        # logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'

    def get_image_anno(self, video, frame, type):
        r"""
        :param video: The video to be used to get the image and box pair
        :param frame: The frame to be used in the video
        :return: image, image_box
        """
        image, image_anno = video[frame]
        image_name = video.img_names[frame]
        ext = image_name.split('.')[-1]
        search_img_name = image_name.split('/')[-1].split('.')[0] + '.' + type+'.'+ext
        image = cv2.imread(os.path.join(self.frame_root, video.name, search_img_name))
        # Do again to handle invalid keys because of box being zero
        if image is None:
            frame = frame-1
            image, image_anno = video[frame]
            image_name = video.img_names[frame]
            ext = image_name.split('.')[-1]
            search_img_name = image_name.split('.')[0] + '.' + type + '.' + ext
            image = cv2.imread(os.path.join(self.frame_root, video.name, ))

        x1, y1, w, h = image_anno
        box = [x1, y1, x1+w, y1+h]
        return image, box

    def get_positive_pair_with_query(self, index):
        r"""
        Method to get a template, search pair from the given video
        referred by the index parameter
        :param index: the index of the video to be used
        :return: (template_image, template_box), (search_image, search_box), query
        """
        video_name = self.video_names[index]
        video = self.videos[video_name]
        frames = video.get_frames()
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - DEFAULT_FRAME_RANGE, 0)
        right = min(template_frame + DEFAULT_FRAME_RANGE, len(frames)-1) + 1
        search_range = list(np.arange(left, right))
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video, template_frame, type='template'), \
            self.get_image_anno(video, search_frame, type='search'), video.get_query()

    def get_random_target_with_query(self, index=-1):
        r"""
        Method to get a random frame from a random video
        :param index: the index of the video to be used for selecting the random frame
        :return: (random_image, random_image_box), query
        :return:
        """
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.video_names[index]
        video = self.videos[video_name]
        frames = video.get_frames()
        random_frame = np.random.randint(0, len(frames))
        return self.get_image_anno(video, random_frame), video.get_query()

    def __len__(self):
        return self.num


class TrackingDataset(Dataset):
    def __init__(self, name, loader, root, frame_root):
        super(TrackingDataset, self).__init__()

        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')


        # Adapt base dataset to tracking dataset loader
        sub_dataset = loader(name=name, dataset_root=root)
        tracking_sub_dataset = TrackingDatasetAdapter(sub_dataset, frame_root=frame_root)
        self.dataset = tracking_sub_dataset
        self.anchor_target = AnchorTarget()
        self.template_aug = Augmentation(
            cfg.DATASET.TEMPLATE.SHIFT,
            cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,
            cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR
        )
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR
        )


    def __len__(self):
        return self.dataset.num

    @staticmethod
    def save_image(template_image, template_box, search_image, search_box, suffix=""):
        r"""
        Utility method to save the template and search images with the box around the area
        :param template_image: Image of the template frame
        :param template_box: Box (x1, y1, x2, y2) of template frame image
        :param search_image: Image of the search frame
        :param search_box: Box (x1, y1, x2, y2) of search frame image
        :return:
        """
        cv2.rectangle(template_image, (int(template_box.x1), int(template_box.y1)),
                      (int(template_box.x2), int(template_box.y2)), (0, 255, 0), 1)
        cv2.rectangle(search_image, (int(search_box.x1), int(search_box.y1)),
                      (int(search_box.x2), int(search_box.y2)), (0, 255, 0), 1)

        cv2.imwrite('template'+suffix+'.png', template_image)
        cv2.imwrite('search'+suffix+'.png', search_image)

    def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    def pos_s_2_bbox(self, pos, s):
        return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]

    def crop_like_SiamFC(self, image, bbox, context_amount=0.5, exemplar_size=127, instance_size=255, padding=(0, 0, 0)):
        target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
        target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        wc_z = target_size[1] + context_amount * sum(target_size)
        hc_z = target_size[0] + context_amount * sum(target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        d_search = (instance_size - exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        z = self.crop_hwc(image, self.pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
        x = self.crop_hwc(image, self.pos_s_2_bbox(target_pos, s_x), instance_size, padding)
        return z, x

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __getitem__(self, index):

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()


        template, search, query = self.dataset.get_positive_pair_with_query(index)

        # get image
        template_image = template[0]
        search_image = search[0]

        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # get bounding box
        #template_box = BBOX(template[1][0], template[1][1], template[1][2]+template[1][0], template[1][3]+template[1][1])
        #search_box = BBOX(search[1][0], search[1][1], search[1][2]+search[1][0], search[1][3]+search[1][1])

        # Uncomment below line if you want to visualize the template and search with box before augmentation for validating
        self.save_image(template_image, template_box, search_image, search_box)
        template_image, template_box = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search_image, search_box = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)
        # Uncomment below line if you want to visualize the template and search with box after augmentation for validating
        self.save_image(template_image, template_box, search_image, search_box, suffix=AUGMENTATION_IMAGE_SUFFIX)
        cls, delta, delta_weight, overlap = self.anchor_target(
            search_box, cfg.TRAIN.OUTPUT_SIZE, neg)
        #print(np.sum(cls==-1))
        print(np.sum(cls==1))
        #print(np.sum(cls==0))
        if np.sum(cls == 1) == 0:
            print(index)
        template = template_image.transpose((2, 0, 1)).astype(np.float32)
        search = search_image.transpose((2, 0, 1)).astype(np.float32)
        # return {
        #     'template': template,
        #     'search': search,
        #     'template_box': np.array(template_box),
        #     'search_box': np.array(search_box),
        #     'query': query,
        # }
        return {
            'template': template,
            'search': search,
            'label_cls': cls,
            'label_loc': delta,
            'label_loc_weight': delta_weight,
            'bbox': np.array(search_box)
        }


if __name__ == '__main__':
    dataset_name = "OTB" # (LaSOT/OTB)
    dataset_root = "/Users/tusharkumar/PycharmProjects/tracking_lang/ln_data/OTBSiamRPNData"

    track_dataset = TrackingDataset(name="OTB_tune_51", loader=OTBDataset,
                                    root="/Users/tusharkumar/PycharmProjects/tracking_lang/ln_data/OTB100/",
                                    frame_root="/Users/tusharkumar/PycharmProjects/tracking_lang/ln_data/OTBSiamRPNData")
    track_dataset[1]
    exit(0)
    for idx, data in enumerate(track_dataset):
        batch_template_image = data['template']

