"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from torchvision.transforms import transforms
from transformers import set_seed
from diffuvqa.rounding import denoised_fn_round
from diffuvqa.vqa_datasets import load_data_vqa

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuvqa.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)

torch.multiprocessing.set_sharing_strategy('file_system')

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def create_argparser():
    defaults = dict(model_path='', step=2500, out_dir='', top_p=0)
    decode_defaults = dict(split='test', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():

    args = create_argparser().parse_args()

    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    num_steps = args.diffusion_steps

    betas =  betas_for_alpha_bar(num_steps, lambda t: 1 - np.sqrt(t + 0.0001),)
    alphas = 1 - betas  # α = 1 - β
    alphas = torch.from_numpy(alphas)
    
    alphas_prod = torch.cumprod(alphas, 0) 
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) 
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
  
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
           alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape == \
           one_minus_alphas_bar_sqrt.shape

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args=args)
    state_dict = torch.load(args.model_path, map_location="cuda")
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(th.device("cuda"))

    tokenizer = load_tokenizer(args)
    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size,
        embedding_dim=args.hidden_dim,
        _weight=model.word_embedding.weight.clone().cuda()
    ).eval().requires_grad_(False)                                                            

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    transform = transforms.Compose([
        transforms.Resize((args.image_resolution, args.image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ## load data
    print(args.batch_size)
    data_test = load_data_vqa(batch_size=args.batch_size, seq_len=args.seq_len, args=args, model_emb=model_emb.cpu(),
                               transform=transform, split=args.split, loaded_vocab=tokenizer, loop=False)

    start_t = time.time()

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.jsonl")

    print("out_path:", out_path)
    print("batch_size:", args.batch_size)
    all_text_data = []
    all_image_data = []

    try:
        for image, cond in data_test:
            cond['input_q_id'] = cond['input_q_id'].to(th.device("cuda"))
            cond['input_ids'] = cond['input_ids'].to(th.device("cuda"))
            all_text_data.append(cond)
            all_image_data.append(image.to(th.device("cuda")))
    except StopIteration:
        print('### End of reading iteration...')

    model_emb.to(th.device("cuda"))

    text_iterator = iter(all_text_data)
    image_iterator = iter(all_image_data)

    for image, cond in zip(image_iterator, text_iterator):

        if not cond:
            continue                                           

        input_ids_x = cond.pop('input_ids').to(th.device("cuda"))
        input_ids_a = cond.pop('input_a_id').to(th.device("cuda"))
        input_emb = model.get_embeds(input_ids_a)
        # qid = cond.pop('qid')
        # print(qid)
        # img_id = cond.pop('img_id')
        # print(img_id)
        # print(input_ids_x)
        input_ids_mask = cond.pop('input_mask').to(th.device("cuda"))
        image_name = cond.pop('image_name')
        # print("input_ids_mask: ", input_ids_mask)
        # print("input_ids_mask.shape: ", input_ids_mask.shape)

        # x_start_mean, _ = model.get_ddpm_inputs_mask(image, cond)
        fuse_feats, _ = model.get_ddpm_input(image, cond)  
        f = torch.cat([fuse_feats, fuse_feats], dim=1)
        print(fuse_feats.shape)
        x_start = torch.cat([fuse_feats, input_emb], dim=1)
        # input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(th.device("cuda"))
        # x_start = th.where(input_ids_mask == 0, x_start_mean, x_start)
        noise = th.randn_like(x_start)
        if args.use_noising_f:
            print("noising f")
            noise = alphas_bar_sqrt[num_steps - 1] * f + one_minus_alphas_bar_sqrt[num_steps - 1] * noise

        x_noised = th.where(input_ids_mask == 0, x_start, noise)  

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap =1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps // args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap
        )

        sample = samples[-1]  
        #
        a_shape = sample.size(1) // 2
        sample = sample[:, a_shape:, :]
        print(sample.shape)
        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)  # th.topk = get_knn

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []
        qid_lst = []
        img_id_lst = []

        print(cands.indices)
        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            # len_x = args.seq_len * args.batch_size - th.sum(input_mask).item()
            seq = seq.to(th.device("cpu"))
            len_x = args.seq_len
            tokens = tokenizer.decode_token(seq)
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            # len_x = args.seq_len - sum(input_mask).item()
            seq = seq.to(th.device("cpu"))
            len_x = args.seq_len
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        fout = open(out_path, 'a')
        for (recov, ref, src, image_name) in zip(word_lst_recover, word_lst_ref, word_lst_source, image_name):
            print(json.dumps(
                {"image_name": image_name, "question": src, "reference_answer": ref, "generate_answer": recov}),
                  file=fout)
        fout.close()
        # break
        #
        # for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
        #     print(json.dumps(
        #         {"question": src, "reference_answer": ref, "generate_answer": recov}),
        #           file=fout)
        # fout.close()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
