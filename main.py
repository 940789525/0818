from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists

import torch
import torch.nn.functional as F
import torch.nn as nn
from tvr.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from tvr.dataloaders.data_dataloaders import DATALOADER_DICT
from tvr.dataloaders.dataloader_msrvtt_retrieval import MSRVTTDataset
from tvr.models.modeling import VTRModel, AllGather
from tvr.models.optimization_adamw import AdamW, get_cosine_schedule_with_warmup
from tvr.utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

from tvr.utils.comm import is_main_process, synchronize
from tvr.utils.logger import setup_logger
from tvr.utils.metric_logger import MetricLogger

from scipy.special import softmax

allgather = AllGather.apply

global logger

def get_args(description='CLIP + LoRA for Text-Video Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0)
    parser.add_argument("--do_eval", type=int, default=0)

    parser.add_argument("--datatype", default="msrvtt", type=str)
    parser.add_argument('--anno_path', type=str, default='data/MSR-VTT/anns')
    parser.add_argument('--video_path', type=str, default='data/MSR-VTT/videos')
    parser.add_argument('--pretrained_path', type=str, default="your_path")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--clip_lr', type=float, default=6e-4, help='base LR for group 0 (LoRA)')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="If warmup_steps==0, use proportion*max_steps.")
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_val', type=int, default=128)

    parser.add_argument('--max_words', type=int, default=32)
    parser.add_argument('--max_frames', type=int, default=12)
    parser.add_argument('--video_framerate', type=int, default=1)

    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--distributed", default=0, type=int)

    parser.add_argument('--n_display', type=int, default=50)
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    parser.add_argument("--base_encoder", default="ViT-B/32", type=str)
    parser.add_argument("--init_model", default=None, type=str, required=False)
    parser.add_argument('--lora_dim', type=int, default=8)

    # ====== 新增：学习率调度相关 ======
    parser.add_argument(
        "--sched", type=str, default="none",
        choices=["none", "linear", "cosine"],
        help="LR scheduler type; 'none' keeps LR constant."
    )
    # 兼容别名（可不传）
    parser.add_argument("--lr_scheduler", type=str, default=None,
                        choices=[None, "linear", "cosine"])

    parser.add_argument(
        "--max_steps", type=int, default=0,
        help="Total training steps. If 0, use len(train_loader)*epochs."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0,
        help="Warmup steps. If 0, use warmup_proportion*max_steps."
    )
    # ====== 新增结束 ======

    args = parser.parse_args()
    return args

# ---- 放在 main.py 顶部任意位置（import 之后）----
class DummyScheduler:
    """始终可用的“空”调度器，接口兼容 get_last_lr/step。"""
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def step(self):
        pass
    def get_last_lr(self):
        # 与 torch scheduler 对齐：返回所有 param_group 当前 lr
        return [g['lr'] for g in self.optimizer.param_groups]
    def state_dict(self):
        return {}
    def load_state_dict(self, state):
        pass

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if torch.cuda.is_available():
        torch.distributed.barrier()
    logger.info("local_rank: {} world_size: {}".format(args.local_rank, args.world_size))

    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and world_size parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.world_size, args.batch_size_val, args.world_size))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def build_model(args):
    model = VTRModel(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device)
    return model


def build_dataloader(args):
    ## ####################################
    # dataloader loading
    ## ####################################
    tokenizer = ClipTokenizer()
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if isinstance(test_length, int):
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    elif len(test_length) == 2:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %dv %dt", test_length[0], test_length[1])
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d %d", len(test_dataloader[0]), len(test_dataloader[1]))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %dv %dt", val_length[0], val_length[1])

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", len(train_dataloader) * args.epochs)
    else:
        train_dataloader, train_sampler = None, None

    return test_dataloader, val_dataloader, train_dataloader, train_sampler





def prep_optimizer(args, model, _max_steps=None, local_rank=None):
    """
    训练：
      - LoRA（名字含 "TVPt"）
      - 轻量 CLS 跨帧注意力 tem_mix_*（q/k/v/out、ln、gate、temp_raw、rel_bias）
      - 空域位置门控 spatial_pos_gate
      - ☆ TimeRouter（frame_proj/patch_proj/score_mlp/budget_head）
    其余参数全部冻结。
    """
    # ---------- 学习率与 weight decay ----------
    base_lr = getattr(args, "clip_lr", None) or getattr(args, "learning_rate", 6e-4)
    wd = getattr(args, "weight_decay", 0.0)

    # ---------- 名称关键字 ----------
    lora_key = "TVPt"
    tem_mix_linear_keys = ("tem_mix_q", "tem_mix_k", "tem_mix_v", "tem_mix_out")
    tem_mix_ln_keys     = ("tem_mix_ln",)
    tem_misc_keys       = ("tem_mix_gate", "tem_rel_bias")
    tem_temp_key        = "tem_mix_temp_raw"
    always_keys         = ("spatial_pos_gate",)

    # ☆ TimeRouter 相关
    tr_root_key         = "time_router"
    tr_proj_keys        = (f"{tr_root_key}.frame_proj", f"{tr_root_key}.patch_proj")
    tr_head_keys        = (f"{tr_root_key}.score_mlp", f"{tr_root_key}.budget_head")

    # ---------- 统一冻结 ----------
    for _, p in model.named_parameters():
        p.requires_grad_(False)

    # ---------- 打开需要训练的参数 ----------
    for n, p in model.named_parameters():
        if (
            (lora_key in n)
            or any(k in n for k in tem_mix_linear_keys + tem_mix_ln_keys + tem_misc_keys + (tem_temp_key,) + always_keys)
            or any(k in n for k in (tr_root_key,))  # ☆ 放开 TimeRouter
        ):
            p.requires_grad_(True)

    # ---------- 分组 ----------
    lora_params, tem_linear, tem_ln, tem_misc, tem_temp, always_params = [], [], [], [], [], []
    tr_proj_params, tr_head_params = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if lora_key in n:
            lora_params.append(p)
        elif any(k in n for k in tem_mix_linear_keys):
            tem_linear.append(p)
        elif any(k in n for k in tem_mix_ln_keys):
            tem_ln.append(p)
        elif any(k in n for k in tem_misc_keys):
            tem_misc.append(p)
        elif tem_temp_key in n:
            tem_temp.append(p)
        elif any(k in n for k in always_keys):
            always_params.append(p)
        elif any(k in n for k in tr_proj_keys):
            tr_proj_params.append(p)
        elif any(k in n for k in tr_head_keys):
            tr_head_params.append(p)

    groups = []
    if lora_params:
        groups.append({'params': lora_params,  'lr': base_lr,        'weight_decay': wd})
    if tem_linear:
        groups.append({'params': tem_linear,   'lr': base_lr * 8.0,  'weight_decay': 0.0})
    if tem_ln:
        groups.append({'params': tem_ln,       'lr': base_lr * 4.0,  'weight_decay': 0.0})
    if tem_temp:
        groups.append({'params': tem_temp,     'lr': base_lr * 2.0,  'weight_decay': 0.0})
    if tem_misc:
        groups.append({'params': tem_misc,     'lr': base_lr * 8.0,  'weight_decay': 0.0})
    if always_params:
        groups.append({'params': always_params,'lr': base_lr * 4.0,  'weight_decay': 0.0})

    # ☆ TimeRouter 分两组：投影/打分
    if tr_proj_params:
        groups.append({'params': tr_proj_params,'lr': base_lr * 1.0, 'weight_decay': 0.0})  # frame/patch proj
    if tr_head_params:
        groups.append({'params': tr_head_params,'lr': base_lr * 4.0, 'weight_decay': 0.0})  # score_mlp / budget_head

    if not groups:
        groups = [{'params': [], 'lr': base_lr, 'weight_decay': wd}]

    optimizer = AdamW(groups, betas=(0.9, 0.98), eps=1e-8)

    # ---------- 简化 scheduler：若未指定则使用 Dummy ----------
    class DummyScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self._last_lr = [g.get('lr', 0.0) for g in self.optimizer.param_groups]
        def step(self):
            self._last_lr = [g.get('lr', 0.0) for g in self.optimizer.param_groups]
        def get_last_lr(self):
            return self._last_lr
        def state_dict(self):
            return {'_last_lr': self._last_lr}
        def load_state_dict(self, sd):
            if '_last_lr' in sd:
                self._last_lr = list(sd['_last_lr'])

    scheduler = None
    sched_name = getattr(args, "lr_scheduler", None) or getattr(args, "sched", None)
    warmup_steps = int(getattr(args, "warmup_steps", 0))
    try:
        if sched_name in ("linear", "cosine"):
            from transformers import (
                get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
            )
            total_steps = _max_steps if _max_steps is not None else getattr(args, "max_steps", 0)
            total_steps = int(total_steps) if total_steps else 0
            if total_steps > 0:
                scheduler = (get_linear_schedule_with_warmup if sched_name == "linear"
                             else get_cosine_schedule_with_warmup)(optimizer, warmup_steps, total_steps)
        if scheduler is None:
            scheduler = DummyScheduler(optimizer)
    except Exception:
        scheduler = DummyScheduler(optimizer)

    # ---------- DDP 包装 ----------
    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )

    # 打印参与训练的关键名
    try:
        named = model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()
        trainable = [n for n, p in named if p.requires_grad]
        print("== Newly enabled trainables (count={}) ==".format(len(trainable)))
        for n in sorted(trainable):
            if any(k in n for k in ("TVPt", "tem_mix", "tem_rel_bias", "spatial_pos_gate", "time_router")):
                print("  " + n)
    except Exception:
        pass

    return optimizer, scheduler, model







def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(args.output_dir, "{}.pth".format(type_name))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def prompt_save_model(epoch, args, model, type_name=""):
    assert "Not Implement" == 0
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(args.output_dir, "{}.pth".format(type_name))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def reduce_loss(loss, args):
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                scheduler, global_step, max_steps, val_dataloader):
    global logger
    global best_score
    global best_score_list
    global meters
    global sim_matrix_num
    global sim_name_list

    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0

    end = time.time()
    logit_scale = 0
    for step, batch in enumerate(train_dataloader, start=1):
        global_step += 1
        data_time = time.time() - end

        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        text_ids, text_mask, video, video_mask, inds, idx = batch
        loss = model(text_ids, text_mask, video, video_mask, idx, global_step)

        optimizer.zero_grad()
        
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        # with torch.autograd.detect_anomaly():
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule

        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.module.clip.logit_scale.exp().item()
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.clip.logit_scale.exp().item()

        batch_time = time.time() - end
        end = time.time()

        reduced_l = reduce_loss(loss, args)
        meters.update(time=batch_time, data=data_time, loss=float(reduced_l))

        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if (global_step % log_step == 0 or global_step == 1) and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "epoch: {epoch}/{max_epoch}",
                        "iteration: {iteration}/{max_iteration}",
                        "{meters}",
                        "lr: {lr}",
                        "logit_scale: {logit_scale:.2f}"
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    max_epoch=args.epochs,
                    iteration=global_step,
                    max_iteration=max_steps,
                    meters=str(meters),
                    lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(scheduler.get_last_lr())))]),
                    logit_scale=logit_scale,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if (global_step % (log_step * 3) == 0)  or global_step == 1:
            max_R1 = eval_epoch(args, model, val_dataloader, args.device)
            if args.local_rank == 0:
                for list_idx in range(sim_matrix_num):
                    if best_score_list[list_idx] < max_R1[list_idx]:
                        best_score_list[list_idx] = max_R1[list_idx]
                    logger.info("The R1 is: {:.4f}\t| {:.4f}\tin {}".format(max_R1[list_idx], best_score_list[list_idx],sim_name_list[list_idx]))

                if best_score < max(max_R1):
                    best_score = max(max_R1)
                    output_model_file = save_model(epoch, args, model, type_name="best")
                logger.info("The best R1 is: {:.4f} at all".format(best_score))

            synchronize()
            model.train()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def eval_epoch(args, model, test_dataloader, device):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
    # ----------------------------
    # 1. cache the features
    # ----------------------------
    batch_cls, batch_mask_t = [], []
    batch_video_feat, batch_mask_v = [], []
    batch_ids = []

    with torch.no_grad():
        tic = time.time()

        sim_matrix = []

        logger.info('[start] extract')
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            text_ids, text_mask, video, video_mask, inds, _ = batch
            cls, video_feat = model.stage1_eval(text_ids, text_mask, video, video_mask)
            batch_cls.append(cls)
            batch_mask_t.append(text_mask)
            batch_video_feat.append(video_feat)
            batch_mask_v.append(video_mask)
            batch_ids.append(inds)

        torch.distributed.barrier()
        
        batch_ids = allgather(torch.cat(batch_ids, dim=0), args).squeeze()
        
        batch_cls = allgather(torch.cat(batch_cls, dim=0), args)
        batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
        batch_video_feat = allgather(torch.cat(batch_video_feat, dim=0), args)
        batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
        
        batch_cls[batch_ids] = batch_cls.clone()
        batch_mask_t[batch_ids] = batch_mask_t.clone()
        batch_video_feat[batch_ids] = batch_video_feat.clone()
        batch_mask_v[batch_ids] = batch_mask_v.clone()
        
        batch_cls = batch_cls[:batch_ids.max() + 1, ...]
        batch_mask_t = batch_mask_t[:batch_ids.max() + 1, ...]
        batch_video_feat = batch_video_feat[:batch_ids.max() + 1, ...]
        batch_mask_v = batch_mask_v[:batch_ids.max() + 1, ...]
        logger.info('[finish] extract')
        
        logger.info('[start] calculate the similarity')
        with torch.no_grad():
            mini_batch = args.batch_size_val
            sim_matrix = []
            
            batch_cls_split = torch.split(batch_cls, mini_batch)
            batch_mask_t_split = torch.split(batch_mask_t, mini_batch)
            batch_video_feat_split = torch.split(batch_video_feat, mini_batch)
            batch_mask_v_split = torch.split(batch_mask_v, mini_batch)
            
            for cls, text_mask in tqdm(zip(batch_cls_split, batch_mask_t_split)):
                each_row = []
                for video_feat, video_mask in zip(batch_video_feat_split, batch_mask_v_split):
                    logits = model.stage2_eval(cls, text_mask, video_feat, video_mask)
                    logits = logits.cpu().detach().numpy()
                    each_row.append(logits)
                each_row = np.concatenate(tuple(each_row), axis=-1)
                sim_matrix.append(each_row)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        logger.info('[finish] calculate the similarity')
        
        
    logger.info('[start] compute_metrics')
    logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1])) 
    global sim_name_list
    
    max_R1=[]
    list_idx = 0
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    logger.info("Eval {} ...".format(sim_name_list[list_idx]))
    logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['R50'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['R50'], vt_metrics['MR'], vt_metrics['MeanR']))
    max_R1.append(tv_metrics['R1'])

    return max_R1

def main():
    global logger
    global best_score
    global best_score_list
    global meters
    global sim_matrix_num
    global sim_name_list

    sim_name_list = ['base'] 
    sim_matrix_num = len(sim_name_list)

    meters = MetricLogger(delimiter="  ")
    args = get_args()
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('tvr', args.output_dir, args.local_rank)

    args = set_seed_logger(args)

    model = build_model(args)

    test_dataloader, val_dataloader, train_dataloader, train_sampler = build_dataloader(args)
    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        _max_steps = len(train_dataloader) * args.epochs
        inferred_steps = len(train_dataloader) * args.epochs
        if args.max_steps <= 0:
            args.max_steps = inferred_steps

        # 若未显式给 warmup_steps，则由占比推导
        if args.warmup_steps <= 0 and args.warmup_proportion > 0:
            args.warmup_steps = int(args.warmup_proportion * args.max_steps)

        optimizer, scheduler, model = prep_optimizer(
            args, model, args.max_steps, args.local_rank
        )

        max_steps = args.max_steps

        best_score = 0.00001
        best_score_list = [0.00001 for _ in range(sim_matrix_num)]
        best_output_model_file = "None"
        global_step = 0
        for epoch in range(args.epochs):
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            synchronize()
            torch.cuda.empty_cache()
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader,
                                               args.device, args.world_size, optimizer,
                                               scheduler, global_step, max_steps, val_dataloader)
            torch.cuda.empty_cache()

            max_R1 = eval_epoch(args, model, val_dataloader, args.device)
            torch.cuda.empty_cache()
            synchronize()

            if args.local_rank == 0:
                for list_idx in range(sim_matrix_num):
                    if best_score_list[list_idx] < max_R1[list_idx]:
                        best_score_list[list_idx] = max_R1[list_idx]
                    logger.info("The R1 is: {:.4f}\t| {:.4f}\tin {}".format(max_R1[list_idx], best_score_list[list_idx],sim_name_list[list_idx]))

                if best_score < max(max_R1):
                    best_score = max(max_R1)
                    output_model_file = save_model(epoch, args, model, type_name="best")
                logger.info("The best R1 is: {:.4f} at all".format(best_score))

            synchronize()

        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n' + f'training finished with {training_time}' + "*" * 20 + '\n')

        if args.local_rank == 0:
            with open("{}_{}.txt".format(args.output_dir, best_score),'w') as f:
                f.write(' ')

    elif args.do_eval:
        eval_epoch(args, model, test_dataloader, args.device)


if __name__ == "__main__":
    main()
