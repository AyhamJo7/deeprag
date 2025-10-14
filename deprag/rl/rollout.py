from typing import List

import torch
from transformers import PreTrainedTokenizer

from ..data.docstore import DocumentStore
from ..models.agent import DeepRAGAgent
from ..models.dsi import DSI
from .buffers import PPOBuffer
from .rewards import compute_qa_reward


def rollout(
    agent: DeepRAGAgent,
    dsi: DSI,
    queries: List[str],
    tokenizer: PreTrainedTokenizer,
    doc_store: DocumentStore,
    retrieval_token_id: int,
    max_new_tokens: int,
    retrieval_penalty: float,
) -> PPOBuffer:
    """Generates trajectories (rollouts) from the agent in the environment."""

    # Tokenize queries
    query_tensors = tokenizer(
        queries, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(agent.model.device)

    # Generation kwargs for the agent
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Generate responses from the agent
    response_tensors, logits, values = agent.generate(query_tensors, **gen_kwargs)

    # For simplicity, we calculate rewards at the end of the trajectory.
    # A more advanced implementation would calculate rewards at each step.

    # Decode responses
    responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Dummy ground truth for reward calculation
    ground_truth = ["dummy answer"] * len(queries)

    # Compute rewards
    rewards = compute_qa_reward(responses, ground_truth)

    # Penalize for retrieval actions
    retrieval_counts = torch.sum(response_tensors == retrieval_token_id, dim=1)
    rewards -= retrieval_penalty * retrieval_counts.to(rewards.device)

    # The logprobs are calculated from the logits of the generated sequence
    logprobs = torch.log_softmax(logits, dim=-1)
    logprobs = torch.gather(logprobs, 2, response_tensors.unsqueeze(-1)).squeeze(-1)

    # Create and return the buffer
    buffer = PPOBuffer(
        query_tensors=query_tensors,
        response_tensors=response_tensors,
        logprobs=logprobs,
        values=values,
        rewards=rewards.unsqueeze(1).repeat(
            1, values.size(1)
        ),  # Distribute reward over trajectory
    )
    buffer.compute_advantages_and_returns(gamma=0.99, lam=0.95)

    return buffer
