import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from ..configs.config import DeepRAGConfig
from ..data.collators import DSICollator
from ..data.loaders import get_dataset
from ..models.dsi import DSI
from ..utils.logging import get_logger
from ..utils.training import get_optimizer

logger = get_logger(__name__)


def train_dsi(config: DeepRAGConfig):
    """Main function to pre-train the DSI model."""
    logger.info("Starting DSI Pre-training")

    # Load DSI model and tokenizer
    dsi = DSI(config.model)
    dsi.model.to(config.device)

    # Load dataset and collator
    dataset = get_dataset(config.data)
    collator = DSICollator(
        tokenizer=dsi.tokenizer,
        max_source_length=config.data.max_seq_length,
        max_target_length=config.data.max_answer_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        collate_fn=collator,
        num_workers=config.train.num_workers,
    )

    # Optimizer and scheduler
    optimizer = get_optimizer(
        dsi.model,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=config.train.warmup_steps,
        num_training_steps=config.train.max_steps,
    )

    # Training loop
    dsi.model.train()
    step = 0
    for epoch in range(100): # Loop indefinitely, break by max_steps
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            if step >= config.train.max_steps:
                break

            batch = {k: v.to(config.device) for k, v in batch.items()}
            loss = dsi.train_step(batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                dsi.model.parameters(), config.train.gradient_clipping
            )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % config.train.logging_steps == 0:
                logger.info(f"Step {step}, Loss: {loss.item():.4f}")

            step += 1
        if step >= config.train.max_steps:
            break

    logger.info("Finished DSI Pre-training")
