# Frequently Asked Questions (FAQ)

### Q: Why use a Differentiable Search Index (DSI) instead of BM25 or FAISS?

A: Traditional retrievers are not differentiable, meaning they cannot be trained end-to-end with the generator. By using a DSI, we can propagate gradients from the final task loss all the way back through the retriever, allowing the retriever to learn what documents are most useful for the generator.

### Q: How does the agent learn *when* to retrieve?

A: The decision to retrieve is modeled as an action in a reinforcement learning problem. The agent receives a positive reward for task accuracy (e.g., answering a question correctly) but a negative reward (penalty) for each retrieval action. By optimizing for the total cumulative reward, the agent learns to balance the benefit of retrieving information against its cost.

### Q: Is this efficient? It seems like calling a T5 model to retrieve is slow.

A: The DSI is indeed more computationally intensive than a traditional retriever at inference time. However, the key advantage is the potential for significantly higher accuracy and the ability to learn a much more efficient retrieval policy. The goal is to perform far fewer, but much higher-quality, retrievals.

### Q: Can I use a different model for the agent or DSI?

A: Yes. The framework is designed to be modular. You can replace the agent or DSI with any Hugging Face compatible model by changing the model configuration in the `deprag/configs/model` directory.
