# DeepRAG Architecture

DeepRAG is designed as a modular, end-to-end trainable system that learns not only *what* to generate but also *when* to retrieve information. This document outlines the core components and their interactions.

## Core Components

The system is composed of three main parts:

1.  **Differentiable Search Index (DSI)**: The retriever.
2.  **DeepRAG Agent**: The generator and policy model.
3.  **MDP Environment**: The reinforcement learning setup.

### 1. Differentiable Search Index (DSI)

-   **Model**: The DSI is an encoder-decoder model (based on T5) that is trained to map a natural language query to a unique document identifier (doc ID).
-   **Function**: Instead of searching an external index, the DSI's parameters *are* the index. It learns a semantic representation of the document store.
-   **Training**: It is pre-trained using a supervised, contrastive objective on `(query, doc_id)` pairs. During joint training, it receives gradients from the agent's end-to-end task loss.

### 2. DeepRAG Agent

-   **Model**: The agent is a large language model (e.g., T5, Llama) augmented with a value head for PPO training.
-   **Action Space**: The agent's vocabulary is extended with a special `[RET]` token. At each decoding step, the agent can either:
    -   Generate a regular content token.
    -   Generate the `[RET]` token to trigger a retrieval.
-   **Policy**: The decision to retrieve is the agent's learned policy. The goal is to learn to retrieve only when necessary, minimizing cost while maximizing accuracy.

### 3. MDP Environment & RL Loop

We frame the generation process as a Markov Decision Process (MDP):

-   **State**: The current state is the sequence of tokens generated so far, including any previously retrieved documents.
-   **Action**: Generate a content token or the `[RET]` token.
-   **Transition**: When `[RET]` is generated, the DSI is queried with the preceding text. The retrieved document is appended to the context, and generation continues.
-   **Reward**: The reward function is a combination of:
    -   **Task Performance**: A proxy for the final task metric (e.g., Exact Match for QA).
    -   **Retrieval Cost**: A negative penalty for each `[RET]` action to encourage efficiency.

-   **Training**: We use Proximal Policy Optimization (PPO) to train the agent's policy, optimizing for the cumulative reward.

## Training Strategy

Training proceeds in three phases:

1.  **DSI Pre-training**: The DSI is trained on a large corpus of `(query, doc_id)` pairs to learn the basic mapping.
2.  **Agent Supervised Warm-start (Optional)**: The agent can be fine-tuned via imitation learning on trajectories with oracle `[RET]` placements.
3.  **Joint RL Finetuning**: The DSI and agent are trained jointly. The agent learns its retrieval policy via PPO, and the DSI is fine-tuned based on the gradients that flow back from the agent's generation loss.
