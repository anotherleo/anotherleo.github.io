# QNN

## Deep Neural Networks(DNN)

A DNN consists of (1) an input layer, (2) multiple hidden layers, and (3) an output layer. 

A DNN with $d$ layers is a non-linear multivariate function $\mathcal{N}: \mathbb{R}^n \to \mathbb{R}^s$.

- Input: $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{x} = \mathbf{x}^1$.
- Hidden layer: $\mathbf{x}^i = \phi(\mathbf{W}^i\mathbf{x}^{i-1} + \mathbf{b}^i)$.
  - Activation function: $\phi$, e.g. $\text{ReLU}(x) = \text{max}(x, 0)$.
  - Weight matrix: $\mathbf{W}^i$, $2 \le i \le d$.
  - Bias vector: $\mathbf{b}^i$, $2 \le i \le d$.
- Output: $\mathcal{N}(\mathbf{x}) = \mathbf{x}^d$.
- Notations: 
  - $n_1 = n, n_2, \dots, n_d = s$ 

## Quantization Scheme

### Yedi Zhang et al.

**Quantization Configuration.** A quantization configuration $\mathcal{C}$ is a tuple $\langle \tau, Q, F \rangle$ , where $Q$ and $F$ are the total bit size and the fractional bit size allocated to a value, respectively, and $\tau \in \{+, ±\}$ indicates if the quantized value is unsigned or signed.

**Example.** Given a real number $x \in \mathbb{R}$ and a quantization configuration $\mathcal{C} = \langle \tau, Q, F \rangle$ , its quantized integer counterpart $\hat{x}$ and the fixed-point counterpart $\tilde{x}$ under the symmetric uniform quantization scheme are: $\hat{x} = \text{clamp}( \lfloor 2^F \cdot x \rceil,\mathcal{C}^{\text{lb}}, \mathcal{C}^{\text{ub}} )$ and $\tilde{x} = \hat{x}/2^F$, where: 

- $\mathcal{C}^{\text{lb}}, \mathcal{C}^{\text{ub}}$
  $$
  \mathcal{C}^{\text{lb}} = 
  \begin{cases}
  0, & \tau = + \\
  -2^{Q-1}, & \text{otherwise}
  \end{cases}
  $$
  
  $$
  \mathcal{C}^{\text{ub}} = 
  \begin{cases}
  2^{Q} - 1, & \tau = + \\
  2^{Q-1} - 1, & \text{otherwise}
  \end{cases}
  $$
  
  
  
- $\lfloor \cdot \rceil$ is the round-to-nearest integer operator

- The clamping function $\text{clamp}(x, a, b)$ with a lower bound $a$ and an upper bound $b$
  $$
  \text{clamp}(x, a, b) = 
  \begin{cases}
  	a, & \text{if } x < a; \\
  	x, & \text{if } a \le x \le b; \\
  	b, & \text{if } x > b.
  \end{cases}
  $$



**Definition (Quantized Neural Network).** Given quantization configurations for the weights, biases, output of the input layer and each hidden layer as $\mathcal{C}_w = \langle \tau_w, Q_w, F_w \rangle$ , $\mathcal{C}_b = \langle \tau_b, Q_b, F_b \rangle$ , $\mathcal{C}_{in} = \langle \tau_{in}, Q_{in}, F_{in} \rangle$ , $\mathcal{C}_h = \langle \tau_h, Q_h, F_h \rangle$, the quantized version (i.e., QNN) of a DNN $\mathcal{N}$ with $d$ layers is a function $\widehat{\mathcal{N}}: \mathbb{Z}^n \to \mathbb{R}^s$ such that $\widehat{\mathcal{N}} = \hat{l}_d \circ \hat{l}_{d−1} \circ \dots \circ \hat{l}_1$. Then, given a quantized input $\hat{\textbf{x}} \in \mathbb{Z}^n$, the output of the QNN $\hat{\textbf{y}} = \mathcal{N} (\hat{\textbf{x}})$ can be obtained by the following recursive computation: 

- Input layer $ \hat{l}_1 : \mathbb{Z}^n → \mathbb{Z}^{n_1}$ is the identity function; 

- Hidden layer $\hat{l}_i : \mathbb{Z}^{n_{i-1}} → \mathbb{Z}^{n_{i}}$ for $2 \le i \le d − 1$ is the function such that for each $j \in [n_i]$, 
  $$
  \hat{\textbf{x}}^i_j = \text{clamp}(\lfloor 2^{F_i} \widehat{\textbf{W}}^i_{j,:} \cdot \hat{\textbf{x}}^{i−1} + 2^{F_h−F_b} \hat{\textbf{b}}^i_j  \rceil, 0, \mathcal{C}^{\text{ub}}_h),
  $$
  where $F_i$ is $F_h − F_w − F_{in}$ if $i = 2$, and $−F_w$ otherwise; 

- Output layer $\hat{l}_d : \mathbb{Z}^{n_{d-1}} \to \mathbb{R}^s$ is the function such that 
  $$
  \hat{\textbf{y}} = \hat{\textbf{x}}^d = \hat{l}_d(\hat{\textbf{x}}^{d−1}) =  2^{−F_w} \widehat{\textbf{W}}^d \hat{\textbf{x}}^{d−1} + 2^{F_h−F_b} \hat{\textbf{b}}^d.
  $$

where for every $2 \leq i \leq d$ and $k \in [n_{i−1}]$, $\widehat{\textbf{W}}^i_{j,k} = \text{clamp}(\lfloor 2^{F_w} \widehat{\textbf{W}}^i_{j,k} \rceil, \mathcal{C}^{\text{lb}}_w, \mathcal{C}^{\text{ub}}_w)$ is the quantized weight and $\hat{\textbf{b}}^i_{j} = \text{clamp}(\lfloor 2^{F_b} {\textbf{b}}^i_{j} \rceil , \mathcal{C}^{\text{lb}}_b, \mathcal{C}^{\text{ub}}_b)$ is the quantized bias.

Note that for $1 \le i \le n$, 
$$
\hat{\textbf{x}}_i = \text{clamp}( \lfloor 2^F \cdot \textbf{x}_i \rceil,\mathcal{C}^{\text{lb}}, \mathcal{C}^{\text{ub}} ) \qquad \mathbf{y}_i = 2^{-F_h} \hat{\mathbf{y}}_i
$$

### Pei Huang et al. 

The quantization operation is a mapping from a real number $γ$ to an integer $q$ of the form
$$
\text{Quant: } q = Round( \frac { \gamma } { s } + z ) , \quad
\text{Dequant: } \gamma = s ( q - z ) ,
$$

- The constant $s$ (for “scale”) is an arbitrary real number. 
- The constant $z$ (for “zero point”) is the integer corresponding to the quantized value $q$ when $γ = 0$. 

**Matrix multiplication.** Suppose we have three $N × N$ matrices of real numbers, where the third matrix is equal to the product of the first two matrices. 

- $r^{(i,j)}_α$: entries of these 3 matrices, where $α ∈ \{1, 2, 3\}$ and $0 ≤ i, j ≤ N − 1$. 
- $q^{(i,j)}_α$: quantized entries.
- $(s_α, z_α)$: quantization parameters.

By quantization scheme, we have: 
$$
s_{3}(q_{3}^{(i,j)}-z_{3})=\sum_{k=0}^{N-1}s_{1}(q_{1}^{(i,k)}-z_{1})s_{2}(q_{2}^{(k,j)}-z_{2}),
$$
*i.e.*, 
$$
q_{3}^{(i,j)}
=
Round\left(
	z_{3}+\frac{s_{1}s_{2}}{s_{3}}\sum_{k=0}^{N-1}(q_{1}^{(i,k)}-z_{1})(q_{2}^{(k,j)}-z_{2})
\right)
.
$$
**Calculation of a single layer.** The transformation is $\mathbf{y} := \text{ReLU} (\mathbf{W} \mathbf{x} + \mathbf{b})$. Its quantized version $\mathbf{y}_q := g (\mathbf{x}_q, \mathbf{W}_q, \mathbf{b}_q)$ can be calculated through: 
$$
\begin{array}{rl}
(\text{i}) & {\,\hat{y}_{0}^{j}:=z_{y}+ \displaystyle \frac{s_{w}^{j}s_{x}}{s_{y}}\sum_{i}(w_{q}^{(i,j)}-z_{w}^{j})(x_{q}^{i}-z_{x})+b_{q}^{j}}\\
(\text{ii}) &{\,\hat{y}_{1}^{j}:=Round(\hat{y}_{0}^{j})}\\
(\text{iii}) &{\,\hat{y}_{2}^{j}:=Clip(\hat{y}_{1}^{j},lb,ub)}\\
(\text{iv}) &{\,y_{q}^{j}:=\operatorname*{max}(\hat{y}_{2}^{j},z_{y})}
\end{array}
$$
?> 注意到这里用$Clip$函数把ReLU函数给编码进去了。

- $z^j_w$: the zero point of the weights corresponding to the $j$th neuron.
- $lb$ and $ub$: the smallest and largest values, respectively, that can be represented by our quantized integer type.
- The $Clip$ function returns the value within $[lb, ub]$ closest to its input.



## Quantization Error Bound

### Yedi Zhang et al.

**Definition (Quantization Error Bound).** Given a DNN $\mathcal{N} : \mathbb{R}^n \to \mathbb{R}^s$, the corresponding QNN $\widehat{\mathcal{N}} : \mathbb{Z}^n \to \mathbb{R}^s$, a quantized input $\hat{\mathbf{x}} \in \mathbb{Z}^n$, a radius $r \in \mathbb{N}$ and an error bound $\epsilon \in \mathbb{R}$. The QNN $\widehat{\mathcal{N}}$ has a quantization error bound of $\epsilon$ w.r.t. the input region $R(\hat{\mathbf{x}}, r) = \{\hat{\mathbf{x}}' \in \mathbb{Z}^n \ |\ \Vert \hat{\mathbf{x}}' − \hat{\mathbf{x}}\Vert_{\infty} \leq r\}$ if for every $\hat{\mathbf{x}}' \in R(\hat{\mathbf{x}}, r)$, we have $||2^{−F_h} \mathcal{N} (\hat{\mathbf{x}}' ) − \mathcal{N} ({\mathbf{x}}')\|_{\infty} < \epsilon$ , where $\mathbf{x}' = \hat{\mathbf{x}}' /(\mathcal{C}^{\text{ub}}_{in} − \mathcal{C}^{\text{lb}}_{in})$.

**Example (Classification Tasks).** Given a DNN $\mathcal{N}$, a corresponding QNN $\widehat{\mathcal{N}}$, a quantized input $\hat{x}$ which is classified to class $g$ by the DNN $\mathcal{N}$, a radius $r$ and an error bound $\epsilon$, the quantization error bound property $P(\mathcal{N} , \widehat{\mathcal{N}} , \hat{x}, r, \epsilon)$ for a classification task can be defined as follows:
$$
\bigwedge_{\hat{\mathbf{x}}' \in R(\hat{\mathbf{x}}, r)} (|2^{-F_h} \widehat{\mathcal{N}}(\hat{\mathbf{x}}')_g - \mathcal{N}({\mathbf{x}}')_g | < \epsilon) \wedge (\mathbf{x}' = \hat{\mathbf{x}}' /(\mathcal{C}^{\text{ub}}_{in} − \mathcal{C}^{\text{lb}}_{in}))
$$
