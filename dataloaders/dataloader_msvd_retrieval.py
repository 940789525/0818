from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from dataloaders.rawvideo_util import RawVideoExtractor
import torch

class MSVD_DataLoader(Dataset):
    """MSVD dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            mask_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.mask_path = mask_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")
        caption_file = os.path.join(self.data_path, "raw-captions.pkl")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_id in video_ids:
            assert video_id in captions
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Paire: {}".format(len(self.sentences_dict)))

        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)
        res = np.zeros((len(choice_video_ids), self.max_frames*2, 1, 3,
                        self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)
        mv = np.zeros((len(choice_video_ids), self.max_frames*2, 1, 3,
                       self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)
        mask = np.zeros((len(choice_video_ids), self.max_frames*2, 1, 1,
                       self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]
            full_video_dir = os.path.join(self.mask_path, video_id)
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path, full_video_dir)
            selected_rows = raw_video_data['video']
            selected_res = raw_video_data['residual']
            selected_mv = raw_video_data['motion_vector']
            selected_mask = raw_video_data['mask_mv'] 

            # L x T x 3 x H x W
            raw_video_slice = self.rawVideoExtractor.process_raw_data(selected_rows)
            raw_res_slice = self.rawVideoExtractor.process_raw_data(selected_res)
            raw_mv_slice = self.rawVideoExtractor.process_raw_data(selected_mv)
            raw_mask_slice = self.rawVideoExtractor.process_raw_data(selected_mask)

            if self.max_frames < raw_video_slice.shape[0]:
                video_slice = raw_video_slice[:self.max_frames, ...]
            else:
                video_slice = raw_video_slice
            if self.max_frames*2 < raw_res_slice.shape[0]:
                res_slice = raw_res_slice[:self.max_frames*2, ...]
            else:
                res_slice = raw_res_slice
            if self.max_frames*2 < raw_mv_slice.shape[0]:
                mv_slice = raw_mv_slice[:self.max_frames*2, ...]
            else:
                mv_slice = raw_mv_slice
            if self.max_frames*2 < raw_mask_slice.shape[0]:
                mask_slice = raw_mask_slice[:self.max_frames*2, ...]
            else:
                mask_slice = raw_mask_slice
            slice_len = video_slice.shape[0]
            res_len = res_slice.shape[0]
            mv_len = mv_slice.shape[0]
            mask_len = mask_slice.shape[0]
            if res_len %2 == 1:
                # 构造一个全 0 的张量，形状为 [1, 1, 3, 224, 224]
                zeros_res = torch.zeros(1, 1, 3, 224, 224)
                zeros_mv = torch.zeros(1, 1, 3, 224, 224)
                zeros_mask = torch.zeros(1, 1, 1, 224, 224)
                # 在 dim=0 上拼接
                res_slice = torch.cat((res_slice, zeros_res), dim=0)
                mv_slice = torch.cat((mv_slice, zeros_mv), dim=0)
                mask_slice = torch.cat((mask_slice, zeros_mask), dim=0)
                res_len += 1
                mv_len += 1
                mask_len += 1

            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[i][:slice_len, ...] = video_slice
                res[i][:res_len, ...] = res_slice
                mv[i][:mv_len, ...] = mv_slice
                mask[i][:mask_len, ...] = mask_slice
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask, res, mv, mask

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask, res, mv,mv_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, res, mv, video_mask, mv_mask
