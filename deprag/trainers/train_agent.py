import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ..configs.config import DeepRAGConfig
from ..data.docstore import DocumentStore
from ..data.loaders import get_dataset
from ..models.agent import DeepRAGAgent
from ..models.dsi import DSI
from ..rl.ppo import DeepRAGPPOTrainer
from ..rl.rollout import rollout
from ..utils.logging import get_logger

logger = get_logger(__name__)


def train_agent(config: DeepRAGConfig):
    """Main function to train the DeepRAG agent with PPO."""
    logger.info("Starting DeepRAG Agent PPO Training")

    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    # Add special tokens if they don't exist

    # Load models
    agent = DeepRAGAgent(config.model).to(config.device)
    dsi = DSI(config.model).to(config.device) # Needed for the environment

    # Load data
    dataset = get_dataset(config.data)
    doc_store = DocumentStore(config.data.doc_store_path)

    # Initialize PPO trainer
    ppo_trainer = DeepRAGPPOTrainer(config.train, agent, tokenizer)

    # Training loop
    for step in tqdm(range(config.train.max_steps), desc="PPO Steps"):
        # Create a batch of queries
        batch = dataset.select(range(config.train.batch_size))
        queries = [item["question"] for item in batch]

        # Generate trajectories
        buffer = rollout(
            agent=agent,
            dsi=dsi,
            queries=queries,
            tokenizer=tokenizer,
            doc_store=doc_store,
            retrieval_token_id=tokenizer.convert_tokens_to_ids(config.model.retrieval_token),
            max_new_tokens=config.model.max_new_tokens,
            retrieval_penalty=config.train.retrieval_penalty,
        )

        # Perform a PPO optimization step
        stats = ppo_trainer.step(
            buffer.query_tensors,
            buffer.response_tensors,
            buffer.rewards.squeeze(1)
        )

        if step % config.train.logging_steps == 0:
            logger.info(f"Step {step}: {stats}")

    logger.info("Finished Agent PPO Training")
