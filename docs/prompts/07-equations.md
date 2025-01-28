# Generating Commonly Used Deep Learning Equations

!!! prompt
    Create a list of the 100 most commonly used
    equations in a college level course in deep learning.

    For each equation, return the following:

    1. The name of the equation in markdown level 4.
    2. Within the body, return the equation in LaTeX format
    3. Surround the equation with dollar signs
    4. For each variable in the equation, add a "where:" that defines each variable
    5. Put blank lines around the list of variables
    6. Wrap the variables names in dollar signs

    Examples:

    #### Slope and Intercept
    $y = m * x + b$

    where:

    - $y$ is the vertical axis
    - $x$ is the horizontal axis
    - $m$ is the slope
    - $b$ is the intercept

    #### Area of a Circle
    $A = 2\pi r^2$
    where:

    - $A$ is the area of the circle
    - $r$ is the radius

## Response

Below are 100 commonly encountered equations in a college-level deep learning course, presented in alphabetical order. Each entry has:

1. A heading in **Markdown level 4**.
2. The equation in **LaTeX**, surrounded by dollar signs.
3. A **where:** section defining each variable, with each variable in dollar signs and blank lines around the list.


# 100 Common Equations in Deep Learning

Below is a curated list of 100 frequently encountered equations in a college-level Deep Learning course. Each entry includes:

- **A name** (in level 4 markdown).
- **The equation** (in LaTeX, surrounded by single dollar signs).
- A **“where:”** section explaining each variable.

The equations are grouped loosely by theme (basic algebra, loss functions, optimization, activations, convolution, RNNs, attention mechanisms, etc.) but may appear in any Deep Learning curriculum.

---

#### 1. Mean Squared Error (MSE)
$L_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y^{(i)} - \hat{y}^{(i)})^2$

where:

- $N$ is the number of samples
- $y^{(i)}$ is the true label for the \(i\)-th sample
- $\hat{y}^{(i)}$ is the predicted value for the \(i\)-th sample

---

#### 2. Cross-Entropy Loss (Binary)
$L_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\Big]$

where:

- $N$ is the number of samples
- $y^{(i)} \in \{0, 1\}$ is the true binary label
- $\hat{y}^{(i)}$ is the predicted probability of the positive class

---

#### 3. Cross-Entropy Loss (Multi-Class)
$L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{c}^{(i)} \log(\hat{y}_{c}^{(i)})$

where:

- $N$ is the number of samples
- $C$ is the number of classes
- $y_{c}^{(i)} \in \{0,1\}$ indicates the true class of sample \(i\)
- $\hat{y}_{c}^{(i)}$ is the predicted probability for class \(c\)

---

#### 4. Softmax Function
$\hat{y}_j = \frac{\exp(z_j)}{\sum_{k=1}^{C}\exp(z_k)}$

where:

- $z_j$ is the logit for class \(j\)
- $C$ is the total number of classes
- $\hat{y}_j$ is the probability assigned to class \(j\)

---

#### 5. Sigmoid (Logistic) Function
$\sigma(z) = \frac{1}{1 + e^{-z}}$

where:

- $z$ is the input or logit
- $\sigma(z)$ is the output between 0 and 1

---

#### 6. Tanh Function
$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

where:

- $z$ is the input
- $\tanh(z)$ outputs values in the range \([-1, 1]\)

---

#### 7. Rectified Linear Unit (ReLU)
$\text{ReLU}(z) = \max(0, z)$

where:

- $z$ is the input
- $\text{ReLU}(z)$ is zero for negative \(z\) and \(z\) itself if \(z > 0\)

---

#### 8. Leaky ReLU
$\text{LeakyReLU}(z) = 
\begin{cases}
z & \text{if } z \ge 0\\
\alpha \, z & \text{if } z < 0
\end{cases}$

where:

- $z$ is the input
- $\alpha$ is a small positive slope (e.g., 0.01) for negative \(z\)

---

#### 9. Weighted Sum of Inputs
$z = \sum_{i=1}^{d} w_i x_i + b$

where:

- $d$ is the number of input features
- $w_i$ is the weight for feature \(x_i\)
- $x_i$ is the \(i\)-th input
- $b$ is the bias term

---

#### 10. Neural Network Output (One Layer)
$\hat{y} = f\bigg(\sum_{i=1}^{d} w_i x_i + b\bigg)$

where:

- $f(\cdot)$ is an activation function
- $w_i$, $x_i$, and $b$ are defined as above

---

#### 11. Gradient Descent (Parameter Update)
$\theta \leftarrow \theta - \eta \,\nabla_\theta L(\theta)$

where:

- $\theta$ represents parameters (weights, biases)
- $\eta$ is the learning rate
- $L(\theta)$ is the loss function
- $\nabla_\theta L(\theta)$ is the gradient of the loss w.r.t. parameters

---

#### 12. Chain Rule (Single-Variable)
$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$

where:

- $f$ and $g$ are functions
- $f'$ and $g'$ are their derivatives

---

#### 13. Chain Rule (Multivariable)
$\frac{\partial}{\partial x_i} L = \sum_{j} \frac{\partial L}{\partial z_j} \frac{\partial z_j}{\partial x_i}$

where:

- $L$ is the loss function
- $z_j$ are intermediate variables dependent on $x_i$

---

#### 14. Momentum Update
\[
\begin{aligned}
v &\leftarrow \beta v + (1-\beta)\nabla_\theta L(\theta), \\
\theta &\leftarrow \theta - \eta\, v
\end{aligned}
\]

where:

- $v$ is the velocity term
- $\beta$ is the momentum hyperparameter (e.g., 0.9)
- $\eta$ is the learning rate
- $\nabla_\theta L(\theta)$ is the gradient

---

#### 15. Nesterov Accelerated Gradient (NAG)
\[
\begin{aligned}
v &\leftarrow \beta v + (1-\beta)\nabla_\theta L(\theta - \beta v),\\
\theta &\leftarrow \theta - \eta\, v
\end{aligned}
\]

where:

- $v$ is the velocity term
- $\beta$ is the momentum factor
- $\theta$ are the parameters
- $\eta$ is the learning rate

---

#### 16. RMSProp Update
\[
\begin{aligned}
E[g^2] &\leftarrow \rho\, E[g^2] + (1-\rho) \, (\nabla_\theta L(\theta))^2, \\
\theta &\leftarrow \theta - \frac{\eta}{\sqrt{E[g^2] + \epsilon}} \nabla_\theta L(\theta)
\end{aligned}
\]

where:

- $E[g^2]$ is the running average of squared gradients
- $\rho$ is the decay rate (e.g., 0.9)
- $\eta$ is the learning rate
- $\epsilon$ is a small constant for numerical stability

---

#### 17. Adam Optimizer
\[
\begin{aligned}
m &\leftarrow \beta_1 m + (1 - \beta_1)\nabla_\theta L(\theta), \\
v &\leftarrow \beta_2 v + (1 - \beta_2)(\nabla_\theta L(\theta))^2, \\
\hat{m} &\leftarrow \frac{m}{1 - \beta_1^t}, \quad \hat{v} \leftarrow \frac{v}{1 - \beta_2^t}, \\
\theta &\leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
\end{aligned}
\]

where:

- $m$ is the first moment estimate
- $v$ is the second moment estimate
- $\beta_1$ and $\beta_2$ are exponential decay rates
- $\eta$ is the learning rate
- $t$ is the current time step
- $\epsilon$ is a small constant

---

#### 18. L1 Regularization
$R_{L1} = \lambda \sum_{j} |w_j|$

where:

- $\lambda$ is the regularization coefficient
- $w_j$ are the weights

---

#### 19. L2 Regularization
$R_{L2} = \frac{\lambda}{2} \sum_{j} w_j^2$

where:

- $\lambda$ is the regularization strength
- $w_j$ are the model’s weights
- The factor $\tfrac{1}{2}$ is often included by convention

---

#### 20. Weighted Cross-Entropy (Class Imbalance)
$L_{\text{WCE}} = -\frac{1}{N} \sum_{i=1}^N \Big[\alpha \, y^{(i)} \log(\hat{y}^{(i)}) + (1-\alpha)\,(1-y^{(i)})\log(1-\hat{y}^{(i)})\Big]$

where:

- $\alpha$ is a weight factor for the positive class
- $y^{(i)} \in \{0,1\}$ is the true label
- $\hat{y}^{(i)}$ is the predicted probability

---

#### 21. Negative Log-Likelihood (NLL)
$L_{\text{NLL}} = -\sum_{i=1}^N \log \, p(y^{(i)} | x^{(i)})$

where:

- $p(y|x)$ is the predicted probability of label \(y\) given input \(x\)
- $N$ is the number of samples

---

#### 22. Normal Equation (Linear Regression)
$\theta = (X^\top X)^{-1} X^\top y$

where:

- $X$ is the design matrix
- $y$ is the vector of targets
- $\theta$ is the vector of parameters

---

#### 23. Coefficient of Determination ($R^2$)
$R^2 = 1 - \frac{\sum_{i=1}^N (y^{(i)} - \hat{y}^{(i)})^2}{\sum_{i=1}^N (y^{(i)} - \bar{y})^2}$

where:

- $y^{(i)}$ is the actual value
- $\hat{y}^{(i)}$ is the predicted value
- $\bar{y}$ is the mean of $y^{(i)}$

---

#### 24. Batch Normalization (Mean)
$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$

where:

- $m$ is the number of samples in a mini-batch
- $x_i$ is the \(i\)-th activation in the batch

---

#### 25. Batch Normalization (Variance)
$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$

where:

- $x_i$ is the \(i\)-th activation in the batch
- $\mu_B$ is the batch mean
- $\sigma_B^2$ is the batch variance

---

#### 26. Batch Normalization (Forward Pass)
$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \, \hat{x}_i + \beta$

where:

- $x_i$ is the input activation
- $\mu_B$, $\sigma_B^2$ are batch statistics
- $\epsilon$ is a small constant
- $\gamma$, $\beta$ are learnable scale and shift parameters

---

#### 27. Dropout (Forward Pass)
$\tilde{h}^{(l)} = \mathbf{m}^{(l)} \odot h^{(l)}$

where:

- $h^{(l)}$ is the activation vector at layer \(l\)
- $\mathbf{m}^{(l)}$ is a random binary mask (each entry ~ Bernoulli(p))
- $\odot$ denotes elementwise multiplication

---

#### 28. 2D Convolution (Without Stride/Padding)
$(I * K)(x,y) = \sum_{u=-r}^{r} \sum_{v=-s}^{s} I(x+u, y+v)\, K(u,v)$

where:

- $I$ is the input image
- $K$ is the kernel
- $r$, $s$ define kernel size offsets

---

#### 29. Transposed Convolution
$\text{ConvTranspose}(z, K) = \sum_{u}\sum_{v} z(u,v)\, \text{Upsample}(K, \text{stride})$

where:

- $z(u,v)$ is an activation map
- $K$ is the kernel
- $\text{stride}$ is the factor by which the output is upsampled

---

#### 30. Max Pooling (2D)
$\text{MaxPool}(x,y) = \max_{(p,q)\in \,\mathcal{R}_{x,y}} I(p,q)$

where:

- $I$ is the input feature map
- $\mathcal{R}_{x,y}$ is the receptive field around location $(x,y)$

---

#### 31. Average Pooling (2D)
$\text{AvgPool}(x,y) = \frac{1}{|\mathcal{R}_{x,y}|}\sum_{(p,q)\in \,\mathcal{R}_{x,y}} I(p,q)$

where:

- $\mathcal{R}_{x,y}$ is the region of pooling
- $|\mathcal{R}_{x,y}|$ is the size of that region

---

#### 32. RNN Hidden State Update
$h_t = f(W_{hh} \, h_{t-1} + W_{xh} \, x_t + b_h)$

where:

- $h_t$ is the hidden state at time \(t\)
- $x_t$ is the input at time \(t\)
- $W_{hh}, W_{xh}$ are weight matrices
- $b_h$ is a bias vector
- $f(\cdot)$ is an activation (e.g., tanh)

---

#### 33. LSTM Input Gate
$i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)$

where:

- $i_t$ is the input gate
- $x_t$ is the current input
- $h_{t-1}$ is the previous hidden state
- $W_{xi}, W_{hi}$ are weight matrices
- $b_i$ is the bias
- $\sigma$ is the sigmoid function

---

#### 34. LSTM Forget Gate
$f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)$

where:

- $f_t$ is the forget gate
- $W_{xf}, W_{hf}$ are weight matrices
- $b_f$ is the bias

---

#### 35. LSTM Output Gate
$o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)$

where:

- $o_t$ is the output gate
- $W_{xo}, W_{ho}$ are weight matrices
- $b_o$ is the bias

---

#### 36. LSTM Cell State
$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \quad \tilde{c}_t = \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c)$

where:

- $c_t$ is the cell state
- $f_t, i_t$ are forget and input gates
- $\tilde{c}_t$ is the candidate cell state
- $\odot$ is elementwise multiplication

---

#### 37. GRU Update Gate
$z_t = \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z)$

where:

- $z_t$ is the update gate
- $x_t$ is the input at time \(t\)
- $h_{t-1}$ is the previous hidden state
- $W_{xz}, W_{hz}$ are weight matrices
- $b_z$ is the bias

---

#### 38. GRU Reset Gate
$r_t = \sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_r)$

where:

- $r_t$ is the reset gate
- $W_{xr}, W_{hr}$ are weight matrices
- $b_r$ is the bias

---

#### 39. GRU Hidden State
$h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t, \quad \tilde{h}_t = \tanh(W_{xh} x_t + r_t \odot (W_{hh} h_{t-1}))$

where:

- $h_t$ is the updated hidden state
- $z_t, r_t$ are the update and reset gates
- $\tilde{h}_t$ is the candidate hidden state

---

#### 40. Attention Score (Dot Product)
$e_{t,s} = h_t^\top \, s_s$

where:

- $h_t$ is the query vector
- $s_s$ is the key vector
- $e_{t,s}$ is the scalar score measuring alignment

---

#### 41. Attention Weights (Softmax)
$\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{k=1}^{S} \exp(e_{t,k})}$

where:

- $e_{t,s}$ is the attention score for query \(t\) and key \(s\)
- $S$ is the total number of key vectors

---

#### 42. Context Vector
$c_t = \sum_{s=1}^{S} \alpha_{t,s}\, v_s$

where:

- $\alpha_{t,s}$ are attention weights
- $v_s$ is the value vector associated with key \(s\)
- $c_t$ is the resulting weighted sum

---

#### 43. Scaled Dot-Product Attention
$\text{Attention}(Q,K,V) = \text{softmax}\bigg(\frac{QK^\top}{\sqrt{d_k}}\bigg)V$

where:

- $Q$ is the matrix of query vectors
- $K$ is the matrix of key vectors
- $V$ is the matrix of value vectors
- $d_k$ is the dimensionality of keys/queries

---

#### 44. Multi-Head Attention
$\text{MHA}(Q,K,V) = \big[\text{head}_1, \dots, \text{head}_h\big]W^O$

where:

- $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
- $W^O$ is the output projection matrix
- $h$ is the number of heads

---

#### 45. Feed-Forward (Transformer)
$\text{FFN}(x) = \max(0, xW_1 + b_1) \, W_2 + b_2$

where:

- $x$ is the input
- $W_1, b_1, W_2, b_2$ are learned parameters
- $\max(0,\cdot)$ indicates ReLU (or another activation)

---

#### 46. Positional Encoding (Sinusoidal)
$\text{PE}(pos, 2i) = \sin\big(pos / 10000^{2i/d_{\text{model}}}\big), \quad \text{PE}(pos, 2i+1) = \cos\big(pos / 10000^{2i/d_{\text{model}}}\big)$

where:

- $pos$ is the position index
- $i$ is the dimension index
- $d_{\text{model}}$ is the model dimensionality

---

#### 47. Perceptron Rule
$\hat{y} = 
\begin{cases}
1 & \text{if } w^\top x + b \ge 0\\
0 & \text{otherwise}
\end{cases}$

where:

- $w$ is the weight vector
- $x$ is the input vector
- $b$ is the bias term

---

#### 48. General Activation Forward
$a = f(z)$

where:

- $z$ is the pre-activation (weighted sum)
- $f(\cdot)$ is an activation function (e.g., sigmoid, ReLU)

---

#### 49. Derivative of ReLU
$\frac{d}{dz}\text{ReLU}(z) =
\begin{cases}
1 & z > 0\\
0 & z \le 0
\end{cases}$

where:

- $z$ is the input

---

#### 50. Derivative of Sigmoid
$\sigma'(z) = \sigma(z)\big(1-\sigma(z)\big)$

where:

- $\sigma(z)$ is the sigmoid function

---

#### 51. Derivative of Tanh
$\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$

where:

- $\tanh(z)$ is the hyperbolic tangent function

---

#### 52. Derivative of Softmax (Vector Form)
$\frac{\partial \hat{y}_j}{\partial z_k} = \hat{y}_j \big(\delta_{jk} - \hat{y}_k\big)$

where:

- $\hat{y}_j$ is the softmax output for class \(j\)
- $z_k$ is the logit for class \(k\)
- $\delta_{jk}$ is the Kronecker delta (1 if \(j=k\), 0 otherwise)

---

#### 53. Derivative of Cross-Entropy w.r.t. Logits (Softmax)
$\frac{\partial L_{\text{CE}}}{\partial z_j} = \hat{y}_j - y_j$

where:

- $\hat{y}_j$ is the predicted probability (softmax output)
- $y_j$ is the one-hot target

---

#### 54. L2 Distance (Euclidean)
$d_{\text{Euclidean}}(x,y) = \sqrt{\sum_{i=1}^{d}(x_i - y_i)^2}$

where:

- $x, y \in \mathbb{R}^d$ are vectors
- $d$ is the number of dimensions

---

#### 55. Manhattan Distance (L1)
$d_{\text{Manhattan}}(x,y) = \sum_{i=1}^d |x_i - y_i|$

where:

- $x, y$ are vectors in \(\mathbb{R}^d\)
- $|\,\cdot\,|$ is the absolute value

---

#### 56. Cosine Similarity
$\text{cos\_sim}(x, y) = \frac{x \cdot y}{\|x\| \, \|y\|}$

where:

- $x \cdot y$ is the dot product
- $\|x\|$ is the norm of $x$

---

#### 57. Dot Product
$x \cdot y = \sum_{i=1}^{d} x_i \, y_i$

where:

- $x, y$ are vectors in \(\mathbb{R}^d\)

---

#### 58. Matrix Multiplication
$(AB)_{ij} = \sum_{k=1}^{n} A_{ik}\, B_{kj}$

where:

- $A$ is a \((m \times n)\) matrix
- $B$ is a \((n \times p)\) matrix
- $(AB)$ is a \((m \times p)\) matrix

---

#### 59. Determinant of a 2×2 Matrix
$\det\begin{pmatrix}
a & b\\
c & d
\end{pmatrix} = ad - bc$

where:

- $a, b, c, d$ are elements of the matrix

---

#### 60. Matrix Inverse (2×2)
$\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} 
= \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a\end{pmatrix}$

where:

- $a, b, c, d$ are elements
- $ad - bc \neq 0$

---

#### 61. Frobenius Norm
$\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} |A_{ij}|^2}$

where:

- $A$ is an \(m \times n\) matrix
- $A_{ij}$ is the element in row \(i\), column \(j\)

---

#### 62. KL Divergence
$D_{\text{KL}}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$

where:

- $P$ and $Q$ are discrete probability distributions
- The sum is over all possible outcomes \(x\)

---

#### 63. Jensen-Shannon Divergence
$D_{\text{JS}}(P\|Q) = \frac{1}{2} D_{\text{KL}}\big(P \| M\big) + \frac{1}{2} D_{\text{KL}}\big(Q \| M\big)$

where:

- $M = \frac{1}{2}(P + Q)$
- $P$ and $Q$ are probability distributions

---

#### 64. Bayes’ Rule
$P(A|B) = \frac{P(B|A)\,P(A)}{P(B)}$

where:

- $A$ and $B$ are events
- $P(A)$, $P(B)$, $P(B|A)$ are known or can be inferred

---

#### 65. Covariance
$\text{Cov}(X,Y) = E[(X - E[X])(Y - E[Y])]$

where:

- $E[\cdot]$ denotes the expectation
- $X, Y$ are random variables

---

#### 66. Pearson Correlation
$\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$

where:

- $\sigma_X$ and $\sigma_Y$ are standard deviations of \(X\) and \(Y\)

---

#### 67. PCA Covariance Eigen-Decomposition
$C = U \Lambda U^\top$

where:

- $C$ is the covariance matrix
- $U$ is the matrix of eigenvectors
- $\Lambda$ is the diagonal matrix of eigenvalues

---

#### 68. Singular Value Decomposition (SVD)
$A = U \Sigma V^\top$

where:

- $A$ is an \((m \times n)\) matrix
- $U$ and $V$ are orthonormal matrices
- $\Sigma$ is the diagonal matrix of singular values

---

#### 69. Gaussian Distribution (1D)
$\mathcal{N}(x \mid \mu,\sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big)$

where:

- $\mu$ is the mean
- $\sigma^2$ is the variance

---

#### 70. Probability Integral (Discrete)
$\sum_{x} P(x) = 1$

where:

- $x$ ranges over all possible outcomes
- $P(x)$ is a probability distribution

---

#### 71. Probability Integral (Continuous)
$\int_{-\infty}^{\infty} p(x)\, dx = 1$

where:

- $p(x)$ is a probability density function

---

#### 72. Exponential Family (General Form)
$p(x|\theta) = h(x) \exp\big(\eta(\theta)^\top T(x) - A(\theta)\big)$

where:

- $h(x)$, $T(x)$ are known functions
- $\eta(\theta)$ is the natural parameter
- $A(\theta)$ is the log-partition function

---

#### 73. Weighted Adjacency (Graph Neural Network)
$h_v^{(l+1)} = \sigma\Bigg(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} \, W^{(l)} h_u^{(l)}\Bigg)$

where:

- $h_v^{(l)}$ is the hidden representation of node \(v\) at layer \(l\)
- $\alpha_{vu}$ is the attention or adjacency weight from node \(u\) to \(v\)
- $W^{(l)}$ is a trainable weight matrix
- $\sigma$ is an activation function

---

#### 74. GNN Aggregation
$h_v^{(l+1)} = \text{AGGREGATE}\big(\{\,h_u^{(l)} : u \in \mathcal{N}(v)\}\big)$

where:

- $\mathcal{N}(v)$ is the set of neighbors of node \(v\)
- $\text{AGGREGATE}(\cdot)$ could be sum, mean, max, or an attention-based mechanism

---

#### 75. RMSProp (Repeated for Emphasis)
$\theta \leftarrow \theta - \frac{\eta}{\sqrt{E[g^2] + \epsilon}} \nabla_\theta L(\theta)$

where:

- $\theta$ is the parameter vector
- $\eta$ is the learning rate
- $E[g^2]$ is the running average of squared gradients
- $\epsilon$ prevents division by zero

---

#### 76. Weighted Combination for Multi-Head
$\text{head}_i = \text{Attention}(QW_i^Q, \; KW_i^K, \; VW_i^V)$

where:

- $Q, K, V$ are query, key, and value matrices
- $W_i^Q, W_i^K, W_i^V$ are parameter matrices for head \(i\)

---

#### 77. Gradient of MSE w.r.t. Weights
$\frac{\partial L_{\text{MSE}}}{\partial w_j} = \frac{1}{N}\sum_{i=1}^N 2 \,(\hat{y}^{(i)} - y^{(i)}) \, x_j^{(i)}$

where:

- $w_j$ is weight \(j\)
- $x_j^{(i)}$ is the \(j\)-th feature of sample \(i\)

---

#### 78. Gradient of Binary Cross-Entropy w.r.t. Weights
$\frac{\partial L_{\text{CE}}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N (\hat{y}^{(i)} - y^{(i)})\, x_j^{(i)}$

where:

- $\hat{y}^{(i)}$ is the predicted probability
- $y^{(i)}$ is the true label

---

#### 79. Logistic Regression Decision Boundary
$\hat{y} = \sigma(w^\top x + b) \implies 
\text{Decision} = \begin{cases}
1 & \text{if } w^\top x + b \ge 0\\
0 & \text{otherwise}
\end{cases}$

where:

- $w, b$ are parameters
- $x$ is the input
- $\sigma$ is the sigmoid function

---

#### 80. Balanced Cross-Entropy
$L_{\text{bal}} = -\frac{1}{N} \sum_{i=1}^N \Big[\beta \, y^{(i)} \log(\hat{y}^{(i)}) + (1-\beta)(1-y^{(i)}) \log(1-\hat{y}^{(i)})\Big]$

where:

- $\beta$ is a factor weighting the positive class
- $y^{(i)}, \hat{y}^{(i)}$ as before

---

#### 81. F1 Score
$\text{F1} = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$

where:

- $\text{precision} = \frac{TP}{TP+FP}$
- $\text{recall} = \frac{TP}{TP+FN}$

---

#### 82. Precision
$\text{precision} = \frac{TP}{TP + FP}$

where:

- $TP$ = true positives
- $FP$ = false positives

---

#### 83. Recall
$\text{recall} = \frac{TP}{TP + FN}$

where:

- $TP$ = true positives
- $FN$ = false negatives

---

#### 84. AUC (Area Under ROC Curve)
$\text{AUC} = \int_0^1 TPR(\text{FPR}^{-1}(x))\, dx$

where:

- $TPR$ is the true positive rate
- $FPR$ is the false positive rate
- The integral is conceptually the area under the ROC curve

---

#### 85. Step Decay (Learning Rate Scheduling)
$\eta_{\text{new}} = \eta_{\text{old}} \cdot \gamma^{\lfloor \frac{\text{epoch}}{k}\rfloor}$

where:

- $\eta_{\text{old}}$ is the current learning rate
- $\eta_{\text{new}}$ is the updated learning rate
- $\gamma \in (0,1)$ is the decay factor
- $k$ is the step size in epochs

---

#### 86. Polynomial Kernel (SVM)
$K(x, x') = (x^\top x' + c)^p$

where:

- $x, x'$ are feature vectors
- $c$ is a constant (often 1)
- $p$ is the polynomial degree

---

#### 87. Radial Basis Function Kernel (RBF)
$K(x, x') = \exp\Big(-\frac{\|x - x'\|^2}{2\sigma^2}\Big)$

where:

- $x, x'$ are feature vectors
- $\sigma$ is the kernel width

---

#### 88. Hinge Loss (SVM)
$L_{\text{hinge}} = \max\big(0, 1 - y_i (w^\top x_i + b)\big)$

where:

- $y_i \in \{-1, +1\}$ is the true label
- $w$ is the weight vector
- $x_i$ is the input
- $b$ is the bias

---

#### 89. Margin (Hard-SVM)
$\text{Margin} = \frac{2}{\|w\|}$

where:

- $w$ is the normal vector to the decision boundary
- $\|w\|$ is the Euclidean norm of $w$

---

#### 90. Q-Learning Update (RL)
$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\Big]$

where:

- $Q(s,a)$ is the current Q-value
- $\alpha$ is the learning rate
- $r$ is the reward
- $\gamma$ is the discount factor
- $s'$ is the next state
- $a'$ is the next action

---

#### 91. Bellman Equation (Value Function)
$V^\pi(s) = \mathbb{E}_{a\sim \pi}\Big[r(s,a) + \gamma V^\pi(s')\Big]$

where:

- $V^\pi(s)$ is the value of state \(s\) under policy \(\pi\)
- $r(s,a)$ is the immediate reward
- $\gamma$ is the discount factor
- $s'$ is the next state

---

#### 92. Actor-Critic Gradient
$\nabla_\theta J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a|s)\, (Q_w(s,a) - b(s))\big]$

where:

- $\pi_\theta$ is the policy (actor)
- $Q_w(s,a)$ is the critic’s estimate of action-value
- $b(s)$ is a baseline to reduce variance

---

#### 93. Softplus Activation
$\text{Softplus}(z) = \log(1 + e^z)$

where:

- $z$ is the input

---

#### 94. ELU (Exponential Linear Unit)
$\text{ELU}(z) =
\begin{cases}
z & z \ge 0,\\
\alpha(e^z - 1) & z < 0
\end{cases}$

where:

- $\alpha$ is a positive constant

---

#### 95. SELU (Scaled ELU)
$\text{SELU}(z) = \lambda \begin{cases}
z & z \ge 0\\
\alpha(e^z - 1) & z < 0
\end{cases}$

where:

- $\alpha$ and $\lambda$ are constants (e.g., \(\alpha \approx 1.673\), \(\lambda \approx 1.051\))

---

#### 96. Knowledge Distillation
$L_{\text{KD}} = \tau^2 \cdot \text{KL}\big(p_\text{teacher}(x;\tau)\,\|\,p_\text{student}(x;\tau)\big)$

where:

- $\tau$ is the temperature
- $p_\text{teacher}$ and $p_\text{student}$ are softmax outputs of teacher and student networks, respectively

---

#### 97. Threshold Function (Binary Step)
$\theta(z) =
\begin{cases}
1 & z \ge 0\\
0 & z < 0
\end{cases}$

where:

- $z$ is the input

---

#### 98. Weighted Hinge Loss
$L_{\text{whinge}} = w_i \,\max\big(0, 1 - y_i (w^\top x_i + b)\big)$

where:

- $w_i$ is the sample weight or class weight
- $y_i \in \{-1, +1\}$ is the true label

---

#### 99. Perplexity (Language Modeling)
$\text{PPL} = \exp\Big(\frac{1}{N}\sum_{i=1}^{N} -\log p(w_i)\Big)$

where:

- $w_i$ are tokens in the sequence
- $p(w_i)$ is the predicted probability of token \(w_i\)
- $N$ is the total number of tokens

---

#### 100. Cycle Consistency Loss (CycleGAN)
$L_{\text{cyc}}(G,F) = \mathbb{E}_{x \sim p(x)}\big[\|F(G(x)) - x\|\big] + \mathbb{E}_{y \sim p(y)}\big[\|G(F(y)) - y\|\big]$

where:

- $G$ is the generator mapping domain \(X \to Y\)
- $F$ is the generator mapping domain \(Y \to X\)
- $x, y$ are samples from domains \(X, Y\)
- $\|\cdot\|$ is a distance measure (e.g., L1 norm)

