# Calculating Tokens Per Second

How quickly a model returns text is a key metric.  Here is a sample
program that calculates the number of tokens per second for Deepseek-r1
running in an Ollama framework.  This test was run on my
local GPU which is a NVIDIA GTX 3090 with 12GB RAM.

To time the performance of a model we do the following

1. Record the time before the model runs with 

```python
start_time = time.time()
```

2. Record the end time and calculate the elapsed time

```python
end_time = time.time()
elapsed_time = end_time - start_time
```

3. Count the total number of tokens in the result and calculate the tokens per second

```python
token_count = len(result.split())
tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
```

## Complete Program

```python
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Define your prompt template.
template = """Question: {question}

Answer: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)

# Initialize the Ollama model.
model = OllamaLLM(model="llama3.1")

# Chain the prompt and the model.
chain = prompt | model

# Record the start time.
start_time = time.time()

# Invoke the chain with your question.
result = chain.invoke({"question": "What are the 50 most imporant concepts in a college level course on deep learning?"})

# Record the end time.
end_time = time.time()
elapsed_time = end_time - start_time

# Count tokens in the output.
# Here we assume tokens are separated by whitespace.
# (For more accurate token counts, consider using the model's specific tokenizer.)
token_count = len(result.split())

# Calculate tokens per second.
tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0

# Print the result and statistics.
print("Result:")
print(result)
print("\nStatistics:")
print(f"Token Count: {token_count}")
print(f"Time Elapsed: {elapsed_time:.2f} seconds")
print(f"Tokens per Second: {tokens_per_second:.2f}")
```

## Result

What a fascinating question!

To tackle this, let's break down the process into manageable steps. Here's a suggested approach:

### **Step 1: Identify Key Topics in Deep Learning**

At a high level, deep learning is a subset of machine learning that focuses on complex models inspired by the structure and function of the brain. The field encompasses various techniques for tasks such as image classification, object detection, speech recognition, natural language processing, and more.

Some key topics to consider when identifying important concepts include:

1. Neural network architectures (e.g., convolutional neural networks, recurrent neural networks)
2. Deep learning frameworks (e.g., TensorFlow, PyTorch)
3. Optimization algorithms (e.g., stochastic gradient descent, Adam)
4. Regularization techniques (e.g., dropout, L1 and L2 regularization)

### **Step 2: Consider Mathematical Foundations**

Deep learning relies heavily on mathematical concepts from linear algebra, calculus, probability theory, and statistics.

Some essential math concepts to cover include:

1. Vector spaces and transformations
2. Matrix operations (e.g., multiplication, inversion)
3. Eigenvalue decomposition and singular value decomposition
4. Probability distributions (e.g., Gaussian, Bernoulli)
5. Expectation and variance

### **Step 3: Explore Practical Aspects of Deep Learning**

This includes the implementation, evaluation, and deployment of deep learning models.

Some important practical concepts to consider:

1. Data preprocessing and augmentation techniques
2. Model selection and hyperparameter tuning
3. Overfitting and regularization strategies
4. Model interpretability and explainability
5. Deployment and scalability in production environments

### **Step 4: Review Advanced Topics and Emerging Trends**

As the field of deep learning continues to evolve, it's essential to stay up-to-date with recent developments.

Some advanced topics and emerging trends include:

1. Transfer learning and pre-trained models
2. Adversarial attacks and defenses
3. Graph neural networks and graph-based techniques
4. Attention mechanisms and transformer architectures
5. Quantum computing and its potential applications in deep learning

### **Step 5: Refine the List of Important Concepts**

Considering the topics identified above, let's prioritize them based on their relevance to a college-level course on deep learning.

Here are the top 50 most important concepts:

### **Mathematical Foundations (1-10)**

1. Vector spaces and transformations
2. Matrix operations (e.g., multiplication, inversion)
3. Eigenvalue decomposition and singular value decomposition
4. Probability distributions (e.g., Gaussian, Bernoulli)
5. Expectation and variance
6. Calculus basics (e.g., gradients, Hessians)
7. Linear algebra review (e.g., determinants, eigenvectors)
8. Probability and statistics review
9. Optimization algorithms (e.g., gradient descent)
10. Regularization techniques (e.g., L1, L2)

### **Neural Network Architectures (11-20)**

11. Convolutional neural networks (CNNs)
12. Recurrent neural networks (RNNs)
13. Long short-term memory (LSTM) networks
14. Gated recurrent units (GRUs)
15. Residual connections and skip connections
16. Autoencoders and variational autoencoders
17. U-Net architectures for image segmentation
18. Transformers and self-attention mechanisms
19. Graph neural networks (GNNs)
20. Other specialized architectures (e.g., capsule networks)

### **Deep Learning Frameworks and Tools (21-25)**

21. TensorFlow and Keras APIs
22. PyTorch and Lightning-PyTorch
23. Deep learning frameworks for GPU acceleration (e.g., cuDNN)
24. Model serving and deployment tools (e.g., Docker, Kubernetes)
25. Deep learning software development kits (SDKs) and libraries

### **Practical Aspects of Deep Learning (26-35)**

26. Data preprocessing and augmentation techniques
27. Model selection and hyperparameter tuning
28. Overfitting and regularization strategies
29. Model interpretability and explainability
30. Deployment and scalability in production environments
31. Model evaluation metrics (e.g., accuracy, precision)
32. Common pitfalls and debugging techniques
33. Data efficiency and transfer learning strategies
34. Regularization techniques for large models
35. Distributed training and parallelization

### **Advanced Topics and Emerging Trends (36-50)**

36. Transfer learning and pre-trained models
37. Adversarial attacks and defenses
38. Graph neural networks and graph-based techniques
39. Attention mechanisms and transformer architectures
40. Quantum computing and its potential applications in deep learning
41. Exponential family distributions and link functions
42. Causal inference and counterfactual reasoning
43. Generative models (e.g., GANs, VAEs)
44. Time series analysis with LSTM networks
45. Text classification and sentiment analysis
46. Image recognition and object detection
47. Speech recognition and natural language processing
48. Reinforcement learning and deep Q-networks (DQN)
49. Multi-agent systems and distributed decision-making
50. Explainability techniques for complex models

Of course, this list is not exhaustive, but it should give you a solid starting point for creating a comprehensive college-level course on deep learning.

How's that?

## Inference Statistics

**Token Count:** 687<br/>
**Time Elapsed:** 13.33 seconds<br/>
**Tokens per Second:** 51.55<br/>