from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import pdb
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from modules.modeling import CLIP4ClipPreTrainedModel, show_log, update_attr, check_attr
# from modules.modeling_DL import DistillationHead
from modules.loss_modules import DistillationLoss,margin_infonce_loss
from modules.modeling_teacher_only import TeacherOnlyModel

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class EstimatorDenseNetTiny(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(EstimatorDenseNetTiny, self).__init__()

        def Conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
                nn.LeakyReLU(0.1)
            )

        self.conv0 = Conv2D(ch_in, 8, kernel_size=3, stride=1)
        dd = 8
        self.conv1 = Conv2D(ch_in + dd, 8, kernel_size=3, stride=1)
        dd += 8
        self.conv2 = Conv2D(ch_in + dd, 6, kernel_size=3, stride=1)
        dd += 6
        self.conv3 = Conv2D(ch_in + dd, 4, kernel_size=3, stride=1)
        dd += 4
        self.conv4 = Conv2D(ch_in + dd, 2, kernel_size=3, stride=1)
        dd += 2
        self.predict_flow = nn.Conv2d(ch_in + dd, ch_out, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = torch.cat((self.conv0(x), x), 1)
        x = torch.cat((self.conv1(x), x), 1)
        x = torch.cat((self.conv2(x), x), 1)
        x = torch.cat((self.conv3(x), x), 1)
        x = torch.cat((self.conv4(x), x), 1)
        return self.predict_flow(x)

class UpsampleUpdatingModel2(nn.Module):
    def __init__(self, dim, mode='mv'):
        super().__init__()
        self.mode = mode
        # self.backbone = SidedataModel(mode=mode)
        self.channel_weight_predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        in_dim = {
            'mv': 2, 'res': 3
        }[mode]


        self.spatial_module = EstimatorDenseNetTiny(in_dim + dim * 2, 1)
        self.channel_module = EstimatorDenseNetTiny(in_dim + dim * 2, dim)

    def forward(self, i_features, p_motions, p_features):
        """
            i_features: (100, 512, 1, 1)
            p_motions: (100, 2, 224, 224)
        Returns:
        """
        p_motions_resized = F.interpolate(p_motions, size=p_features.shape[-2:], mode='bilinear', align_corners=False)

        channel_weight = self.channel_module(torch.cat([p_motions_resized, p_features, i_features], dim=1))
        weight = self.channel_weight_predictor(F.adaptive_avg_pool2d(channel_weight, 1).flatten(1))  # (300, 256)
        # weight = self.channel_weight_predictor(F.adaptive_max_pool2d(channel_weight, 1).flatten(1))  # (300, 256)
        temp = 2
        weight = weight / temp

        i_features = i_features * weight.unsqueeze(-1).unsqueeze(-1)  # (bn gop) c h w
        spatial_weight = self.spatial_module(torch.cat([p_motions_resized, p_features, i_features], dim=1))
        spatial_weight = F.softmax(spatial_weight.view(*spatial_weight.shape[:2], -1), dim=-1).view_as(spatial_weight)

        spatial_weight = spatial_weight / temp
        i_features = (i_features * spatial_weight).sum(dim=(2, 3))  # (bn gop) c
        p_features = i_features + F.adaptive_avg_pool2d(p_features, 1).flatten(1)  # (bn gop) c
        # tempc = 0.2
        # i_features = (1-tempc) * i_features + tempc * (i_features * weight.unsqueeze(-1).unsqueeze(-1))  # (bn gop) c h w
        # spatial_weight = self.spatial_module(torch.cat([p_motions_resized, p_features, i_features], dim=1))
        # spatial_weight = F.softmax(spatial_weight.view(*spatial_weight.shape[:2], -1), dim=-1).view_as(spatial_weight)
        # temps = 0.2
        # i_features = ((1-temps) * i_features + temps * (i_features * spatial_weight)).sum(dim=(2, 3))  # (bn gop) c
        # p_features = i_features + F.adaptive_avg_pool2d(p_features, 1).flatten(1)  # (bn gop) c

        return p_features

class XCLIP(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(XCLIP, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")
        self.new_added_modules = getattr(task_config, "new_added_modules", [None, ])
        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        num_words = task_config.max_words
        num_frames = self.task_config.max_frames

        # recommend set True
        self.use_original_clip_for_frame_features = True    

        # for coarse-grained constrast weights
        self.global_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)

        # for cross-grained constrast weights
        self.word_logit_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.frame_logit_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)        

        # for fine-grained constrast weights
        self.local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
        self.frame_mat_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        self.word_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)

        # self.loss_fct = CrossEn()
        # --- 在这里实例化所有损失函数 ---
        # self.loss_main_fn = InfoNCELoss()
        # self.loss_feat_fn = FeatureDistillationLoss()
        # self.loss_logits_fn = LogitsDistillationLoss(temperature=4.0)

        # self.loss_fct = DistillationLoss(
        #     final_alpha=1.0,
        #     final_beta=1.0,
        #     warmup_proportion=0.4,    
        #     increase_proportion=0.2,
        #     l1_margin=0.1    
        # )

        self.apply(self.init_weights)
        dim = 512
        # self.mv_module = UpsampleUpdatingModel2(dim, mode='mv')
        # self.res_module = UpsampleUpdatingModel2(dim, mode='res')

        # self.DL = DistillationHead(dim,num_frames,2)

        self.teacher = TeacherOnlyModel(512,12,2)
    def forward(self, input_ids, token_type_ids, attention_mask, video, res, mv, video_mask=None,mv_mask=None,train_proportion=0.0):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        res = torch.as_tensor(res).float()
        mv = torch.as_tensor(mv).float()
        mv_mask = torch.as_tensor(mv_mask).float()

        b, pair, bs, ts, channel, h, w = video.shape
        res_b, res_pair, res_bs, res_ts, res_channel, res_h, res_w = res.shape
        mv_b, mv_pair, mv_bs, mv_ts, mv_channel, mv_h, mv_w = mv.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        res = res.view(res_b * res_pair * res_bs * res_ts, res_channel, res_h, res_w)
        mv = mv.view(mv_b * mv_pair * mv_bs * mv_ts, mv_channel, mv_h, mv_w)
        mv_mask = mv_mask.view(res_b * res_pair * res_bs * res_ts, 1, res_h, res_w)
        video_frame = bs * ts

        # [bs, 1, dim], [bs, num_words, dim], [bs, num_frames, dim]
        (sequence_output, seq_features), visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask, 
                                                                video, video_mask, shaped=True, video_frame=video_frame)
        # v_features_teacher, v_features_student = self.DL(visual_output,video,mv,res,mv_mask,video_mask)
        # Pvisual_output = self.get_pvisual_output(visual_output, video_mask, res, mv, mv_mask, shaped=True)
        v_features_teacher = self.teacher(visual_output,video,mv,res,mv_mask,video_mask)


        if self.training:
            # loss = 0.
            # sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, Pvisual_output, attention_mask,
            #                             video_mask, shaped=True, loose_type=self.loose_type)
            # # sim_matrix1 = self.get_Ploose_similarity(sequence_output, Pvisual_output, video_mask)
            # sim_loss1 = self.loss_fct(sim_matrix)
            # sim_loss2 = self.loss_fct(sim_matrix.T)
            # sim_loss = (sim_loss1 + sim_loss2) / 2
            # loss += sim_loss

            # --- 规范化特征 ---
            # v_features_teacher_norm = F.normalize(v_features_teacher, p=2, dim=-1)
            # v_features_student_norm = F.normalize(v_features_student, p=2, dim=-1)
            # squeezed_sequence_output = sequence_output.squeeze(1)
            # t_features_norm = F.normalize(squeezed_sequence_output, p=2, dim=-1)

             # --- 分别计算三个损失 ---
            # loss_main = self.loss_main_fn(v_features_teacher_norm, t_features_norm)

            # loss_feat = self.loss_feat_fn(v_features_student, v_features_teacher) # MSE通常在非归一化特征上计算

            # loss_logits = self.loss_logits_fn(v_features_student_norm, v_features_teacher_norm, t_features_norm)

            # --- 加权求和得到总损失 ---
            # alpha = 1.0
            # beta = 0.5
            # total_loss = loss_main + alpha * loss_feat + beta * loss_logits
            squeezed_sequence_output = sequence_output.squeeze(1)
            # total_loss = self.loss_fct(v_features_teacher, v_features_student, squeezed_sequence_output, video_mask, train_proportion)

            # return total_loss['loss_total']
            def _masked_mean(features, mask):
                """
                辅助函数：计算带掩码的平均池化，用于将序列特征聚合成单个向量。
                """
                mask_expanded = mask.unsqueeze(-1).expand_as(features)
                # 计算有效帧的特征总和，然后除以有效帧的数量
                return (features * mask_expanded).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            v_teacher_summary = _masked_mean(v_features_teacher, video_mask)
            loss = margin_infonce_loss(
            features_a=v_teacher_summary,
            features_b=squeezed_sequence_output,
            margin=0.0
            )
            return loss
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden, seq_features = self.clip.encode_text(input_ids, return_hidden=True)
        sequence_hidden, seq_features = sequence_hidden.float(), seq_features.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden, seq_features

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_pvisual_output(self, visual_output, video_mask, res, mv, mask_mv, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            res = torch.as_tensor(res).float()
            mv = torch.as_tensor(mv).float()
            res_b, res_pair, res_bs, res_ts, res_channel, res_h, res_w = res.shape
            mv_b, mv_pair, mv_bs, mv_ts, mv_channel, mv_h, mv_w = mv.shape
            res = res.view(res_b * res_pair * res_bs * res_ts, res_channel, res_h, res_w)
            mv = mv.view(mv_b * mv_pair * mv_bs * mv_ts, mv_channel, mv_h, mv_w)
        bs_pair = video_mask.size(0)
        b, fr, t = visual_output.shape
        i_features = visual_output.view(b * fr, t)
        i_features = i_features.unsqueeze(-1).unsqueeze(-1)
        video_frame = fr
        res_feather = self.clip.encode_res(res, video_frame=video_frame).float()
        res_feather = res_feather.view(bs_pair, -1, res_feather.size(-1))
        b, fr, t = res_feather.shape
        res_feather = res_feather.view(b * fr, t)
        res_feather = res_feather.unsqueeze(-1).unsqueeze(-1)

        mv_feather = self.clip.encode_mv(mv, video_frame=video_frame).float()
        mv_feather = mv_feather.view(bs_pair, -1, mv_feather.size(-1))
        b, fr, t = mv_feather.shape
        mv_feather = mv_feather.view(b * fr, t)
        mv_feather = mv_feather.unsqueeze(-1).unsqueeze(-1)

        p_features = self.mv_module(i_features, mv, mv_feather)
        p_features += self.res_module(i_features, res, res_feather)
        bs_pair = video_mask.size(0)
        p_features = p_features.view(bs_pair, -1, p_features.size(-1))
        return p_features

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output, seq_features = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True) # [bs, 1, dim], [bs, num_words, dim]
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)                  # [bs, num_frames, dim]

        return (sequence_output, seq_features), visual_output

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un  # 掩码处理  去除无效的多余帧
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum  # 得到视频特征
        return video_out #[1,512]

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, seq_features, visual_output,Pvisual_output, attention_mask, video_mask, sim_header="meanP"):
        """
            sequence_output: CLS token of text       # [bs, 1, dim]
            seq_features: all tokens of text         # [bs, num_words, dim]
            visual_output: all frames of video       # [bs, num_frames, dim]
        """
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            visual_output_original = visual_output  # 走这个分支  记录原始视频特征
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        # video-level visual feature 
        video_output = visual_output / visual_output.norm(dim=-1, keepdim=True)  # [1,12,512] -> [1,12,512]  最后一个维度归一化？
        video_output = self._mean_pooling_for_similarity_visual(video_output, video_mask)  # 平均池化求视频特征
        video_output = video_output / video_output.norm(dim=-1, keepdim=True)                    # [bs, dim]  再次归一化

        # frame-level visual features       
        if self.use_original_clip_for_frame_features:
            frame_features = visual_output_original / visual_output_original.norm(dim=-1, keepdim=True)                # [bs, num_frames, dim]  帧特征归一化  可能不需要了
        else:
            frame_features = visual_output / visual_output.norm(dim=-1, keepdim=True)                                  # [bs, num_frames, dim]
        # video-level visual feature
        Pvisual_output = Pvisual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            visual_output_original = Pvisual_output
            pass
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = Pvisual_output
            seq_length = Pvisual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=Pvisual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(Pvisual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            Pvisual_output = Pvisual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            Pvisual_output = Pvisual_output.permute(1, 0, 2)  # NLD -> LND
            Pvisual_output = self.transformerClip(Pvisual_output, extended_video_mask)
            Pvisual_output = Pvisual_output.permute(1, 0, 2)  # LND -> NLD
            Pvisual_output = Pvisual_output + visual_output_original

        # video-level visual feature
        Pvideo_output = Pvisual_output / Pvisual_output.norm(dim=-1, keepdim=True) # [1,512] 这里形状有问题，调试进行修改
        Pvideo_output = self._mean_pooling_for_similarity_visual(Pvideo_output, video_mask)
        Pvideo_output = Pvideo_output / Pvideo_output.norm(dim=-1, keepdim=True)                    # [bs, dim]

        # frame-level visual features
        if self.use_original_clip_for_frame_features:
            Pframe_features = visual_output_original / visual_output_original.norm(dim=-1, keepdim=True)                # [bs, num_frames, dim]  压缩域帧特征
        else:
            Pframe_features = Pvisual_output / Pvisual_output.norm(dim=-1, keepdim=True)                                  # [bs, num_frames, dim]

        # sentence-level textual feature
        sentence_output = sequence_output.squeeze(1)   # 句子特征 [bs,1,512]
        sentence_output  = sentence_output / sentence_output.norm(dim=-1, keepdim=True)          # [bs, dim]  归一化 消除dim=1这个维度
        
        # word-level textual features
        word_features = seq_features / seq_features.norm(dim=-1, keepdim=True)                   # [bs, num_words, dim]

        logit_scale = self.clip.logit_scale.exp()  # 一般为100

        if self.training:
            video_output = allgather(video_output, self.task_config)
            Pvideo_output = allgather(Pvideo_output, self.task_config)
            frame_features = allgather(frame_features, self.task_config)
            Pframe_features = allgather(Pframe_features, self.task_config)
            sentence_output = allgather(sentence_output, self.task_config)
            word_features = allgather(word_features, self.task_config)
            torch.distributed.barrier()

        # video-sentence score 
        video_sentence_logits = logit_scale * torch.matmul(torch.matmul(sentence_output, self.global_mat_weight), video_output.t())
        Pvideo_sentence_logits = logit_scale * torch.matmul(torch.matmul(sentence_output, self.global_mat_weight),
                                                           Pvideo_output.t())

        # video-word score
        video_word_logits = logit_scale * torch.sum(torch.matmul(word_features, video_output.t()) \
            * torch.matmul(torch.softmax(torch.matmul(word_features, video_output.t()) / 1e-2, dim=1).permute(0,2,1), self.word_logit_weight).permute(0,2,1), dim=1)
        Pvideo_word_logits = logit_scale * torch.sum(torch.matmul(word_features, Pvideo_output.t()) \
                                                    * torch.matmul(
            torch.softmax(torch.matmul(word_features, Pvideo_output.t()) / 1e-2, dim=1).permute(0, 2, 1),
            self.word_logit_weight).permute(0, 2, 1), dim=1)

        # sentence-frame score 
        sentence_frame_logits = logit_scale * torch.sum(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) \
            * torch.matmul(torch.softmax(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight), dim=-1).t()
        Psentence_frame_logits = logit_scale * torch.sum(torch.matmul(sentence_output, Pframe_features.permute(0, 2, 1)) \
            * torch.matmul(torch.softmax(torch.matmul(sentence_output, Pframe_features.permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight), dim=-1).t()

        # frame-word score
        frame_word_logits = logit_scale * self._attenion_over_fine_grained_sim_matrix(word_features, frame_features)
        Pframe_word_logits = logit_scale * self._attenion_over_fine_grained_sim_matrix(word_features, Pframe_features)

        logits = (video_sentence_logits + video_word_logits + sentence_frame_logits + frame_word_logits) / 4
        Plogits = (Pvideo_sentence_logits + Pvideo_word_logits + Psentence_frame_logits + Pframe_word_logits) / 4
        alpha = 0.4
        logits=(1-alpha)*logits+alpha*Plogits
        return logits  # 损失函数值？？？？ [32,1]  [bs,1]

    def _attenion_over_fine_grained_sim_matrix(self, word_features, frame_features):
        bs_video, num_frames, dim_video = frame_features.shape
        bs_text, num_words, dim_text = word_features.shape
        fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.view(-1, dim_text), self.local_mat_weight), frame_features.view(-1, dim_video).t()).view(bs_text, num_words, bs_video, num_frames)  # [bs_text, num_words, bs_video, num_frames]

        word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        frame_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.frame_mat_weight) * fine_grained_sim_scores, dim=-1)                                             # [bs_text, num_words, bs_video]

        sent2frame_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.frame_mat_weight2) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        video2word_logits = torch.sum(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        return (sent2frame_logits + video2word_logits) / 2

    def get_Ploose_similarity(self, sequence_output, visual_output, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        # frame-level visual features
        frame_features = visual_output / visual_output.norm(dim=-1, keepdim=True)  # [bs, num_frames, dim]

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)


        logit_scale = self.clip.logit_scale.exp()

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            frame_features = allgather(frame_features, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        # sentence-frame score
        sentence_frame_logits = logit_scale * torch.sum(torch.matmul(sequence_output, frame_features.permute(0, 2, 1)) \
            * torch.matmul(torch.softmax(torch.matmul(sequence_output, frame_features.permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight), dim=-1).t()
        # video-sentence score
        video_sentence_logits = logit_scale * torch.matmul(torch.matmul(sequence_output, self.global_mat_weight), visual_output.t())

        # retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        retrieve_logits = (sentence_frame_logits+video_sentence_logits)/2
        return retrieve_logits

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, seq_features, visual_output, Pvisual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])  # 句子掩码 [16,1,32] -> [16,32]
            video_mask = video_mask.view(-1, video_mask.shape[-1])  # 视频掩码 [1,1,12]->[1,12]

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits = self._loose_similarity(sequence_output, seq_features, visual_output,Pvisual_output, attention_mask, video_mask, sim_header=self.sim_header)
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )
        return retrieve_logits, contrastive_direction
    

    def calculate_student_similarity_corrected(self, sequence_output, visual_output, video_mask):
        """
        【修正版】为您的学生模型定制的相似度计算函数。
        
        该函数假定输入的 'visual_output' 已经是通过学生模型处理后的
        最终帧级别特征序列 (v_features_student)，并加入了logit_scale。

        :param self: 模型实例
        :param sequence_output: 文本特征, shape: [batch_size_text, 1, feature_dim]
        :param visual_output: 学生模型输出的视频帧特征序列, shape: [batch_size_video, seq_len, feature_dim]
        :param video_mask: 视频有效帧掩码, shape: [batch_size_video, 1, seq_len]
        :return: 经过缩放的相似度矩阵 logits, shape: [batch_size_text, batch_size_video]
        """
        
        # --- 1. 预处理输入张量，去除多余的维度 ---
        text_features = sequence_output.squeeze(1)
        video_mask = video_mask.squeeze(1)

        # --- 2. 对已有的学生模型输出的帧序列进行掩码平均，得到视频级全局特征 ---
        def masked_mean(features, mask):
            mask_expanded = mask.unsqueeze(-1).expand_as(features)
            return (features * mask_expanded).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)

        video_features_pooled = masked_mean(visual_output, video_mask)
        
        # --- 3. 计算文本和视频的相似度 ---
        text_features_norm = F.normalize(text_features, p=2, dim=-1)
        video_features_norm = F.normalize(video_features_pooled, p=2, dim=-1)
        
        logits = torch.matmul(text_features_norm, video_features_norm.t())
        
        logit_scale = 100.0
        logits = logits * logit_scale
        
        return logits
    

    # def train(self, mode=True):
    #     """
    #     Override the default train() to freeze the BN parameters
    #     :return:
    #     """
    #     super(XCLIP, self).train(mode)
    #     no_clip = self.new_added_modules
    #     logging.info("(model) Freezing ALL the CLIP backbone.")
    #     for name, param in self.clip.named_parameters():
    #         if not any(nd in name for nd in no_clip):
    #             logging.info('Freezing parameters are:{}'.format(name))
    #         else:
    #             logging.info('trainerble parameters are:{}'.format(name))
    #
    #     for name, m in self.clip.named_modules():
    #         if not any(nd in name for nd in no_clip):
    #             if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.Dropout)):
    #                 m.eval()
    #     for m in self.clip.modules():
    #         if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
    #             m.eval()
    #
    # def print_trainable_params_percentage(self):
    #
    #     orig_param_size = sum(p.numel() for p in self.clip.parameters())
    #
    #     def count_parameters(model):
    #         return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    #     trainable_size = count_parameters(self.clip)
    #
    #     percentage = trainable_size / orig_param_size * 100
    #
    #     print(f"Trainable param percentage: {percentage:.2f}% ({trainable_size}/{orig_param_size})")
    #
    #     return percentage