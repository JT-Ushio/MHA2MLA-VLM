import argparse
from transformers import AutoConfig, LlavaForConditionalGeneration
from swift.llm import load_dataset, get_template
from swift.llm import get_model_tokenizer
from tqdm import tqdm
import types
import os
import torch
import numpy as np
import random

from transformers.modeling_utils import load_sharded_checkpoint
from safetensors.torch import load_file
from patching_model_load import patch_model
from patching_llama_stage1 import mha2mla_llama


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_dataloader(train_dataset, template, batch_size=1, shuffle=True):
    from torch.utils.data import Dataset, DataLoader

    class SwiftDataset(Dataset):
        def __init__(self, dataset, template):
            super().__init__()
            self.dataset = dataset
            self.template = template

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            ret = template.encode(self.dataset[index])
            return ret

    dataset = SwiftDataset(train_dataset, template)
    data_collator = template.data_collator

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=shuffle,
    )

    return dataloader


def main(args):
    # init....
    base_model_name_or_path = args.base_model
    stage1_path = args.stage1_path
    dataset_path = args.dataset_path
    output_path = args.output_path

    total_num = args.total_num
    batch_size = args.batch_size
    shuffle = args.shuffle
    low_rank = args.low_rank
    seed = args.seed

    setup_seed(seed)

    train_dataset, _ = load_dataset(dataset_path)

    config = AutoConfig.from_pretrained(stage1_path)
    mha2mla_args = types.SimpleNamespace(**config.mha2mla)
    print("init model")
    mha_model = LlavaForConditionalGeneration(
        config
    ).cuda()
    # patch model and load weight
    print("patch model")
    mla_model_s1, q_idx, k_idx = patch_model(mha_model, mha_model.config.get_text_config(), mha2mla_args)

    single_weight_file = os.path.join(stage1_path, "model.safetensors")
    print("load parameters")
    if os.path.exists(single_weight_file):
        mla_model_s1.load_state_dict(load_file(single_weight_file))
    else:
        load_sharded_checkpoint(mla_model_s1, stage1_path)

    if isinstance(mla_model_s1, LlavaForConditionalGeneration):
        mha2mla_llama(q_idx, k_idx)
    else:
        raise ValueError("unsupported model")

    # load template and dataloader
    base_model, processor = get_model_tokenizer(base_model_name_or_path)
    template = get_template(base_model.model_meta.template, processor)
    del base_model
    template.set_mode("train")
    template.model = mla_model_s1
    dataloader = get_dataloader(train_dataset, template, batch_size=batch_size, shuffle=shuffle)

    model = mla_model_s1.to(dtype=torch.float32).cuda()
    print(model)

    # compute S=XX^T
    class HookContext:
        def __init__(self):
            self.kvproj_inputs = {}
            self.image_mask = None
            self.cnt = 0
            self.image_token_acc = 0
            self.text_token_acc = 0

    hook_ctx = HookContext()

    def save_input_hook_split(module, input, output):
        # import pdb; pdb.set_trace();

        def cal_s(emb):
            tmp = emb.reshape(-1, emb.size(-1))
            tmp = tmp.transpose(0, 1) @ tmp
            return tmp
        if (
                f"{module.layer_id}.img" in hook_ctx.kvproj_inputs.keys()
                and f"{module.layer_id}.text" in hook_ctx.kvproj_inputs.keys()
        ):
            hook_ctx.kvproj_inputs[f"{module.layer_id}.img"] += cal_s(input[0][hook_ctx.image_mask]).detach()
            hook_ctx.kvproj_inputs[f"{module.layer_id}.text"] += cal_s(input[0][~hook_ctx.image_mask]).detach()
        else:
            hook_ctx.kvproj_inputs[f"{module.layer_id}.img"] = cal_s(input[0][hook_ctx.image_mask]).detach()
            hook_ctx.kvproj_inputs[f"{module.layer_id}.text"] = cal_s(input[0][~hook_ctx.image_mask]).detach()

    # def save_input_hook_joint(module, input, output):

    #     def cal_s(emb):
    #         tmp = emb.reshape(-1, emb.size(-1))
    #         tmp = tmp.transpose(0, 1) @ tmp
    #         return tmp

    #     if (
    #             f"{module.layer_id}" in hook_ctx.kvproj_inputs.keys()
    #     ):
    #         hook_ctx.kvproj_inputs[f"{module.layer_id}"] += cal_s(input[0]).detach()
    #     else:
    #         hook_ctx.kvproj_inputs[f"{module.layer_id}"] = cal_s(input[0]).detach()

    save_input_hook = save_input_hook_split

    # register hook
    for layer_id, layer in enumerate(model.language_model.model.layers):
        kvproj_module = layer.self_attn.kv_proj
        kvproj_module.layer_id = layer_id
        kvproj_module.register_forward_hook(save_input_hook)

    # forward
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="testing"):
            batch = {k: v.cuda() for k, v in batch.items()}
            hook_ctx.image_mask = (batch["input_ids"] == config.image_token_index)
            model(**batch)
            hook_ctx.image_token_acc += hook_ctx.image_mask.sum()
            hook_ctx.text_token_acc += (~hook_ctx.image_mask).sum()
            hook_ctx.cnt += batch["input_ids"].size(0)
            if hook_ctx.cnt >= total_num:
                break

    # init split states S
    S = {
        "S": hook_ctx.kvproj_inputs,
        "image_token_acc": hook_ctx.image_token_acc,
        "text_token_acc": hook_ctx.text_token_acc,
        "sample_acc": hook_ctx.cnt
    }

    SVD_weight = {
        "W_up_k_img": {},
        "W_up_k_text": {},
        "W_up_v_img": {},
        "W_up_v_text": {},
        "W_down_img": {},
        "W_down_text": {},
    }

    r = model.config.get_text_config().num_key_value_heads * low_rank
    for layer_id in tqdm(range(config.get_text_config().num_hidden_layers)):
        # image
        input = S["S"][f"{layer_id}.img"] / S["image_token_acc"] / S["sample_acc"]
        Us, Ss, Vs = torch.linalg.svd(input.float(), full_matrices=False)
        Wk = model.language_model.model.layers[layer_id].self_attn.kv_proj.up_k.weight
        Wv = model.language_model.model.layers[layer_id].self_attn.kv_proj.up_v.weight
        W = torch.cat([Wk, Wv])
        Ss_sqrt = torch.diag(torch.sqrt(Ss))
        D = W @ Us @ Ss_sqrt
        Uws, Sws, Vws = torch.linalg.svd(D.float(), full_matrices=False)
        Uws, Sws, Vws = Uws[:, :r], Sws[:r], Vws[:r, :]
        Ss_sqrt_inv = torch.linalg.inv(Ss_sqrt)
        Us_inv = torch.linalg.inv(Us)
        W_up = Uws @ torch.diag(torch.sqrt(Sws))
        W_down = torch.diag(torch.sqrt(Sws)) @ Vws @ Ss_sqrt_inv @ Us_inv
        W_up_k = W_up[: Wk.size(0), :]
        W_up_v = W_up[Wk.size(0):, :]
        SVD_weight["W_up_k_img"][layer_id] = W_up_k.detach().cpu()
        SVD_weight["W_up_v_img"][layer_id] = W_up_v.detach().cpu()
        SVD_weight["W_down_img"][layer_id] = W_down.detach().cpu()
        print(f"[{layer_id} image]: W_up_k.size(), W_up_v.size(), W_down.size()")

        # text
        input = S["S"][f"{layer_id}.text"] / S["text_token_acc"] / S["sample_acc"]
        Us, Ss, Vs = torch.linalg.svd(input.float(), full_matrices=False)
        Wk = model.language_model.model.layers[layer_id].self_attn.kv_proj.up_k.weight
        Wv = model.language_model.model.layers[layer_id].self_attn.kv_proj.up_v.weight
        W = torch.cat([Wk, Wv])
        Ss_sqrt = torch.diag(torch.sqrt(Ss))
        D = W @ Us @ Ss_sqrt
        Uws, Sws, Vws = torch.linalg.svd(D.float(), full_matrices=False)
        Uws, Sws, Vws = Uws[:, :r], Sws[:r], Vws[:r, :]
        Ss_sqrt_inv = torch.linalg.inv(Ss_sqrt)
        Us_inv = torch.linalg.inv(Us)
        W_up = Uws @ torch.diag(torch.sqrt(Sws))
        W_down = torch.diag(torch.sqrt(Sws)) @ Vws @ Ss_sqrt_inv @ Us_inv
        W_up_k = W_up[: Wk.size(0), :]
        W_up_v = W_up[Wk.size(0):, :]
        SVD_weight["W_up_k_text"][layer_id] = W_up_k.detach().cpu()
        SVD_weight["W_up_v_text"][layer_id] = W_up_v.detach().cpu()
        SVD_weight["W_down_text"][layer_id] = W_down.detach().cpu()
        print(f"[{layer_id} text]: W_up_k.size(), W_up_v.size(), W_down.size()")

    # save weights
    torch.save(
        SVD_weight,
        output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--stage1_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--total_num", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--low_rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()


    main(args)

