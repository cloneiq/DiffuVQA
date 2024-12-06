import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import io
import torch 

from diffuvqa.utils import dist_util, logger
from diffuvqa.utils.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from diffuvqa.utils.nn import update_ema
from diffuvqa.step_sample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            learning_steps=0,
            checkpoint_path='',
            gradient_clipping=-1.,
            eval_data=None,
            eval_interval=-1,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.learning_steps = learning_steps
        self.gradient_clipping = gradient_clipping

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self.checkpoint_path = checkpoint_path  # DEBUG **

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            # self._load_optimizer_state()
            frac_done = (self.step + self.resume_step) / self.learning_steps
            lr = self.lr * (1 - frac_done)
            self.opt = AdamW(self.master_params, lr=lr, weight_decay=self.weight_decay)
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():  # DEBUG **
            self.use_ddp = True
            print(dist_util.dev())

            self.ddp_model = self.model
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):

        pass

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    actual_model_path(ema_checkpoint), map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if bf.exists(main_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {main_checkpoint}")
            state_dict = dist_util.load_state_dict(
                actual_model_path(main_checkpoint), map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self, config):
        # while (
        #     not self.learning_steps
        #     or self.step + self.resume_step < self.learning_steps
        # ):
        #     image, cond = next(self.data)
        #     self.run_step(image, cond)
        #     if self.step % self.log_interval == 0:
        #         logger.dumpkvs()
        #     if self.eval_data is not None and self.step % self.eval_interval == 0:
        #         batch_eval, cond_eval = next(self.eval_data)
        #         self.forward_only(batch_eval, cond_eval)
        #         print('eval on validation set')
        #         logger.dumpkvs()
        #     if self.step > 0 and self.step % self.save_interval == 0:
        #         self.save()
        #         # Run for a finite amount of time in integration tests.
        #         if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
        #             return
        #     self.step += 1
        #     print(self.step)
        from tqdm import tqdm
        for epoch in range(self.learning_steps):
            for image, cond in self.data:
                self.run_step(image, cond)
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.eval_data is not None and self.step % self.eval_interval == 0:
                    batch_eval, cond_eval = next(self.eval_data)
                    self.forward_only(batch_eval, cond_eval)
                    print('eval on validation set')
                    logger.dumpkvs()
                if self.step > 0 and self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
                print(f'Epoch: {epoch}, Step: {self.step}')
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, image, cond):
        self.forward_backward(image, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_only(self, image, cond):
        print(self.batch_size)
        with th.no_grad():
            zero_grad(self.model_params)
            for i in range(0, image.shape[0], self.microbatch):
                image = image[i: i + self.microbatch].to(dist_util.dev())
                del cond['image_name']
                cond = {
                    k: v[i: i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= image.shape[0]
                # image = image.to(dist_util.dev())
                # del cond['qid']
                # del cond['img_id']
                micro_cond = {
                    k: v.to(dist_util.dev())
                    for k, v in cond.items()
                }
                # last_batch = (i + self.microbatch) >= image.shape[0]
                t, weights = self.schedule_sampler.sample(image.shape[0], dist_util.dev())
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    image,
                    t,
                    model_kwargs=cond,

                )

                # if last_batch or not self.use_ddp:
                losses = compute_losses()
                # else:
                #     with self.ddp_model.no_sync():
                #         losses = compute_losses()
                loss = (losses["loss"] * weights).mean()
                log_loss_dict(
                    self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()}
                )
                print("eval loss:", loss.detach())

    def forward_backward(self, image, cond):
        zero_grad(self.model_params)
        for i in range(0, image.shape[0], self.microbatch):
            image = image[i: i + self.microbatch].to(dist_util.dev())
            del cond['image_name']
            cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            t, weights = self.schedule_sampler.sample(image.shape[0], dist_util.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                image,
                t,
                model_kwargs=cond,
            )

            losses = compute_losses()
            # else:
            #     with self.ddp_model.no_sync():
            #         losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()
        print("loss", loss.detach())

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def grad_clip(self):
        # print('doing gradient clipping')
        max_grad_norm = self.gradient_clipping  #3.0
        if hasattr(self.opt, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.opt.clip_grad_norm(max_grad_norm)
        # else:
        #     assert False
        # elif hasattr(self.model, "clip_grad_norm_"):
        #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
        #     self.model.clip_grad_norm_(args.max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            th.nn.utils.clip_grad_norm_(
                self.model.parameters(),  #amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            self.grad_clip()
        self._log_grad_norm()
        self._anneal_lr()
        # scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=20000, gamma=0.1)
        self.opt.step()
        # scheduler.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        # cnt = 0
        for p in self.master_params:
            # print(cnt, p) ## DEBUG
            # print(cnt, p.grad)
            # cnt += 1
            if p.grad != None:
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.learning_steps:
            return
        frac_done = (self.step + self.resume_step) / self.learning_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    
    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            print('writing to', bf.join(get_blob_logdir(), filename))
            print('writing to', bf.join(self.checkpoint_path, filename))
            # with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
            #     th.save(state_dict, f)
            with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f:  # DEBUG **
                th.save(state_dict, f)  # save locally
                # pass # save empty

        # save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()), master_params  # DEBUG **
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    if filename[-3:] == '.pt':
        return int(filename[-9:-3])
    else:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def actual_model_path(model_path):
    return model_path
