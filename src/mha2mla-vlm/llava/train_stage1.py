
import torch
import argparse

from arguments import MHA2MLAModelArguments, TrainingArguments
from dataclasses import asdict
from transformers import HfArgumentParser, LlamaModel


from swift.llm.train import SwiftSft
from swift.trainers import Seq2SeqTrainer
from swift.utils import get_logger, get_model_parameter_info
from swift.plugin.optimizer import optimizers_map

from patching_model_load import patch_model
from patching_llama_stage1 import mha2mla_llama
from lr_scheduler import create_constant_with_warmup_decay_optimizer, set_lr_scheduler_args, calculate_max_steps
from utils import freeze_non_attn_weights, convert_ratio_to_steps


logger = get_logger()

def main():
    # load arguments from cfg file only
    cfg_parser = argparse.ArgumentParser()
    cfg_parser.add_argument("--cfg_file", type=str, required=True, help="Path to YAML configuration file")
    cfg = cfg_parser.parse_args()
    
    hf_parser = HfArgumentParser(
        (TrainingArguments, MHA2MLAModelArguments)
    )
    
    # Load from YAML file
    training_args, mha2mla_args = hf_parser.parse_yaml_file(cfg.cfg_file)
    
    if training_args.optimizer == "constant_with_warmup_decay":
        set_lr_scheduler_args(training_args.optimizer_lr_scheduler_kwargs)
        optimizers_map[training_args.optimizer] = create_constant_with_warmup_decay_optimizer
        
    fire(training_args, mha2mla_args)
    
def fire(training_args, mha2mla_args):
    return LlavaMLASFT(training_args, mha2mla_args).main()

class LlavaMLASFT(SwiftSft):
    def __init__(self, training_args, mha2mla_args) -> None:
        super().__init__(training_args)
        self.mha2mla_args = mha2mla_args

    def _mha2mla(self, model):
        model = model.to(torch.float32) # SVD needs float32
        model, q_idx, k_idx = patch_model(model, model.config.get_text_config(), self.mha2mla_args)
        model.config.mha2mla = asdict(self.mha2mla_args)
        if isinstance(model.language_model.model, LlamaModel):
            mha2mla_llama(q_idx, k_idx)
        else:
            raise ValueError("unsupported model")
        logger.info(f"model_info: {model.model_info}")
        model = model.to(dtype=self.args.torch_dtype) 
        return model

    def run(self):
        args = self.args

        # prepare dataset
        train_dataset, val_dataset = self._get_dataset()
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)

        # Convert ratio-based scheduler parameters to absolute steps
        if args.optimizer == "constant_with_warmup_decay" and args.optimizer_lr_scheduler_kwargs:
            total_steps = calculate_max_steps(args, train_dataset)
            logger.info(f"[Training] Total training steps: {total_steps}")
            
            converted_kwargs = convert_ratio_to_steps(args.optimizer_lr_scheduler_kwargs, total_steps)
            args.optimizer_lr_scheduler_kwargs = converted_kwargs
            
            # Re-apply the converted values to the lr_scheduler global config
            set_lr_scheduler_args(converted_kwargs)
            logger.info(f"[LR Scheduler] Updated scheduler args: {converted_kwargs}")

        args.save_args()    # save args

        data_collator = self._get_data_collator()

        self.model = self._mha2mla(self.model)

        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)

        if self.mha2mla_args.peft_train is not None:
            from transformers import LlavaForConditionalGeneration
            freeze_non_attn_weights(self.model.language_model, self.mha2mla_args.peft_train)
            if isinstance(self.model, LlavaForConditionalGeneration):
                for param in self.model.language_model.lm_head.parameters():
                    param.requires_grad = False
            else:
                raise ValueError("unsupport model")

        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)

        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)

if __name__ == "__main__":
    main()

    