from dataclasses import dataclass, field
from typing import Optional, List
import swift.llm
import swift

@dataclass
class TrainingArguments(swift.llm.argument.TrainArguments):
    optimizer_lr_scheduler_kwargs: Optional[str] | Optional[dict] = field(
        default=None,
        metadata={
            "help": "Arguments for optimizer and learning rate scheduler, e.g., {'lr_decay_steps': 1000, 'lr_warmup_steps': 100}"
        }
    )

@dataclass
class MHA2MLAModelArguments:
    peft_train: Optional[str] = field(
        default=None,
        metadata={
            "help": "whether to use peft"
        }
    )
    partial_rope_version: str = field(
        default="high",
        metadata={
            "help": "RoPE version to use for partial RoPE in MLA. Options: 'high', 'low', 'uniform', 'mkl'"
        },
    )
    rope_dim_for_mla: int = field(
        default=0, metadata={"help": "Number of rope dimensions per head"}
    )
    uniform_start_point: int = field(
        default=0,
        metadata={
            "help": "Starting point (only used when partial_rope_version='uniform')"
        },
    )
    qk_tensor_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pre-computed QK tensor file, e.g., 'utils/qk_tensor_135M.pth'"
        },
    )
    svd_init_method: str = field(
        default="none",
        metadata={
            "help": "Method for SVD initialization. Options: 'split' or 'joint' or 'none'"
        },
    )
    low_rank: int = field(
        default=8, metadata={"help": "Rank for low-rank approximation in MLA"}
    )
    is_gqa2mha2mla: bool = field(
        default=False,
        metadata={"help": "if the finetuning is GQA2MHA2MLA"},
    )
    stage1_path: str = field(
        default="none",
        metadata={"help": "path to stage1 checkpoint."},
    )
    svd_init_weight_path: str = field(
        default="none",
        metadata={"help": "path to load svd init weight."},
    )

    def __post_init__(self):
        # Call parent class __post_init__ first to ensure any parent validation happens
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        # Validate partial_rope_version
        valid_rope_versions = ["high", "low", "uniform", "mkl"]
        if self.partial_rope_version not in valid_rope_versions:
            raise ValueError(
                f"partial_rope_version must be one of {valid_rope_versions}, got '{self.partial_rope_version}'"
            )

        # Validate svd_init_method
        valid_svd_methods = ["none", "split", "joint", "only_key", "only_value"]
        if self.svd_init_method not in valid_svd_methods:
            raise ValueError(
                f"svd_init_method must be one of {valid_svd_methods}, got '{self.svd_init_method}'"
            )