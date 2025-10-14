from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    dataset_name: str = "hotpot_qa"
    path: str = "hotpot_qa"
    subset: Optional[str] = None
    max_seq_length: int = 512
    max_answer_length: int = 64
    doc_store_path: str = "data/docstore.jsonl"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"


@dataclass
class ModelConfig:
    name: str = "dsi_small"
    model_name_or_path: str = "t5-small"
    max_seq_length: int = 512
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q", "v"])
    retrieval_token: str = "<RET>"
    doc_start_token: str = "<DOC>"
    doc_end_token: str = "</DOC>"
    max_new_tokens: int = 50
    num_beams: int = 3
    doc_id_prefix: str = "doc-"
    num_doc_ids_per_batch: int = 8
    top_k: int = 5


@dataclass
class PPOConfig:
    p_steps: int = 256
    p_epochs: int = 4
    p_batch_size: int = 16
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    adap_kl_ctrl: bool = True
    gamma: float = 1.0
    lam: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    vf_coef: float = 0.1
    whiten_advantages: bool = True


@dataclass
class TrainConfig:
    task: str = "dsi_pretrain"
    max_steps: int = 50000
    learning_rate: float = 5.0e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    eval_every_steps: int = 1000
    logging_steps: int = 100
    optimizer: str = "AdamW"
    checkpoint_dir: str = "checkpoints"
    save_total_limit: int = 3
    num_workers: int = 4
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward_fn: str = "qa_em"
    retrieval_penalty: float = 0.1
    dsi_learning_rate: Optional[float] = None


@dataclass
class DeepRAGConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"data": "hotpotqa"},
            {"model": "dsi_small"},
            {"train": "dsi_pretrain"},
            {"override hydra/job_logging": "colorlog"},
            {"override hydra/hydra_logging": "colorlog"},
        ]
    )
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"},
            "sweep": {
                "dir": "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}",
                "subdir": "${hydra.job.num}",
            },
        }
    )
    project_name: str = "deeprag"
    seed: int = 42
    device: str = "cuda"
    distributed: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="deeprag_config", node=DeepRAGConfig)
    cs.store(group="data", name="hotpotqa", node=DataConfig)
    cs.store(group="data", name="synthetic", node=DataConfig)
    cs.store(group="model", name="dsi_small", node=ModelConfig)
    cs.store(group="model", name="agent_small", node=ModelConfig)
    cs.store(group="train", name="dsi_pretrain", node=TrainConfig)
    cs.store(group="train", name="agent_ppo", node=TrainConfig)
    cs.store(group="train", name="joint_finetune", node=TrainConfig)
