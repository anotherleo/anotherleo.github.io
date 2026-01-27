# Towards Efficient Verification of Quantized Neural Networks

<center>Pei Huang, Haoze Wu, Yuting Yang, Ieva Daukantas, Min Wu, Yedi Zhang, Clark Barrett*</center>

Quantization replaces floating point arithmetic with integer arithmetic in deep neural network models, providing more efficient on-device inference with less power and memory. In this work, we propose a framework for formally verifying properties of quantized neural networks. Our baseline technique is based on integer linear programming which guarantees both soundness and completeness. We then show how efficiency can be improved by utilizing gradient-based heuristic search methods and also bound-propagation techniques. We evaluate our approach on perception networks quantized with PyTorch. Our results show that we can verify quantized networks with better scalability and efficiency than the previous state of the art.

## Introduction

In recent years, deep neural networks (DNNs) (Goodfellow, Bengio, and Courville 2016) have demonstrated tremendous capabilities across a wide range of tasks (Simonyan and Zisserman 2015; Devlin et al. 2019; Dosovitskiy et al. 2021). However, DNNs have also shown various security and safety issues, e.g., vulnerability to input perturbations (Goodfellow, Shlens, and Szegedy 2015; Huang et al. 2022b,a; Yang et al. 2023). Such issues must be addressed before DNNs can be used in safety-critical scenarios such as autonomous driving (Xu et al. 2017) and medical diagnostics (Ciresan et al. 2012). Formal verification is an established technique which applies mathematical reasoning to ensure the correct behavior of safety-critical systems, and several approaches for applying formal methods to DNNs have been investigated (Huang et al. 2017; Lechner et al. 2022).

Our focus is the verification of quantized neural networks (QNNs). Quantization replaces inputs and parameters represented as 32/64-bit floating point numbers with a lower bitwidth fixed point (e.g., 8-bits) representation (Jacob et al. 2018; Han, Mao, and Dally 2016). QNNs can greatly reduce both memory requirements and computational costs while maintaining competitive accuracy. As a result, they are increasingly being used in embedded applications, including safety-critical applications such as autonomous driving. For instance, 8-bit quantized DNNs have been applied in Tesla’s Full Self-Driving Chip (previously Autopilot Hardware 3.0) (Henzinger, Lechner, and Žikelić 2021; Tes FSD Chip-Tesla). With the increasing popularization and use of QNNs, it is urgent to develop efficient and effective verification techniques for them.

In this work, we propose an efficient verification framework for QNNs with three components, offering different trade-offs between scalability and precision. The baseline approach models neural networks and formal properties as integer linear programming (ILP) problems. ILP is an exact method in the sense that it guarantees both soundness (if it reports the system is safe, then it really is safe) and completeness (if the system really is safe, then it will report that it is safe). Unlike previous work, which focuses on simple models of quantized neural networks, ours is the first formal approach that precisely captures the quantization scheme used in popular deep learning frameworks such as PyTorch/TensorFlow.

Our ILP approach is precise but may encounter scalability issues on larger QNNs. To address this, we also propose a gradient-based method for finding counterexamples. We use a rewriting trick for the non-differentiable round operation, which enables the backward process to cross through the round operation and gives us the desired gradient information. If this method finds a counterexample, then we immediately know that the property does not hold, without having to invoke the ILP solver.

The third component of the framework lies in between the first two. It relies on abstract interpretation-based reasoning to do an incomplete but formal analysis. We extend existing abstract interpretation-based interval analysis methods to support the semantics of “round” and “clip” operations in quantized neural networks. In particular, for the clip operation, we reduce it to a gadget built from two ReLU units. If the abstract interpretation approach succeeds, we know the property holds. Otherwise, the result of the analysis can be used to reduce the runtime of the ILP-based complete method. The overall framework is depicted in Fig. 1.

Based on our framework, we realize an **E**fficient **Q**NN **V**erification system named EQV. We use EQV to verify the robustness of QNNs against bounded input perturbations. Our experimental results show that EQV can scale to networks that are more than twice as large as the largest networks handled by previous approaches. We also show that, compared to the baseline ILP technique, EQV is up to 100 × more efficient for some cases. 

Our contributions can be summarized as the following: 

1. We provide a ILPbased exact verification approach for the QNNs which first precisely captures the quantization scheme used in current popular deep learning frameworks; 
2. We extend existing abstract interpretation-based interval analysis methods to support QNNs; 
3. We design a rewriting trick for the non-differentiable round operation, which enables gradientbased analysis of QNNs; 
4. We implement our approach in a tool, EQV, and demonstrate that it can scale to networks that are twice the size of the largest analyzed by the current state-of-the-art methods, and up to 100 × faster than the baseline ILP method.

## Background and Related Work

Formal DNN verification checks whether a DNN satisfies a property such as the absence of adversarial examples in a given perturbation space. The property is usually depicted by a formal specification, and verifiers aim to provide either a proof of the validity of this property or a counterexample. Researchers have developed a range of verification techniques, mostly for real-valued ReLU networks. 

- **Exact methods** (i.e., sound and complete) can always, in theory, answer whether a property holds or not in any situation. Typical exact methods formalize the verification problem as a Satisfiability Modulo Theories (SMT) problem (Katz et al. 2017; Ehlers 2017; Huang et al. 2017; Jia et al. 2023) or a Mixed Integer Linear Programming (MILP) problem (Cheng, Nührenberg, and Ruess 2017; Fischetti and Jo 2018; Dutta et al. 2018), but their scalability is limited as the problem is NP-hard (Katz et al. 2017). 
- Another typical approach is to use a method that only guarantees soundness, i.e., to improve the scalability at the cost of completeness. **Abstract interpretation** is one such approach. It overapproximates the behavior of the neural network with the hope that the property can still be shown to hold (Wong and Kolter 2018; Weng et al. 2018; Gehr et al. 2018; Zhang et al. 2018; Raghunathan, Steinhardt, and Liang 2018; Mirman, Gehr, and Vechev 2018; Singh et al. 2019). 
- Finally, **heuristic approaches** can be used to search for counterexamples. These techniques are neither sound nor complete but can be effective in practice (Goodfellow, Shlens, and Szegedy 2015; Yang et al. 2022; Serban, Poll, and Visser 2021).

Existing work on DNN verification typically focuses on networks whose parameters are real or floating point numbers. In contrast, relatively little prior work addresses the verification of QNNs. QNN verification presents additional challenges due to the difficulty of **modeling quantization schemes**. And some evidence suggests that it may also be more computationally challenging. For example, Jia et al. (Jia and Rinard 2020) point out that the verification of binarized neural networks (which can be regarded as 1-bit quantized neural networks) has exhibited even worse scalability than real-valued neural network verification.

?> 这种computationally challenging是否能用理论刻画？

In the last two years, some work has started to focus on the verification of QNNs. Henzinger et al. (Henzinger, Lechner, and Žikelić 2021) provide an SMT-based method to encode the problem as a formula in the SMT theory of bit-vectors. Mistry et al. (Mistry, Saha, and Biswas 2022) and Zhang et al. (Zhang et al. 2022) propose using MILP and ILP to model the QNN verification problem. All of these methods pioneer new directions for QNNs but are applied only to small models using simple quantization schemes. None of them can directly support the sophisticated quantization schemes used in real deep learning frameworks.

?> 这篇文章对现实quantization scheme的支持是一大亮点。


## Preliminaries

The quantization operation is a mapping from a real number $γ$ to an integer $q$ of the form
$$
\text{Quant: } q = Round( \frac { \gamma } { s } + z ) , \quad
\text{Dequant: } \gamma = s ( q - z ) ,
\tag{1}
$$
for some constants $s$ and $z$. Equation 1 is the quantization scheme, and the constants $s$ and $z$ are quantization parameters. The constant $s$ (for “scale”) is an arbitrary real number. The constant $z$ (for “zero point”) is the integer corresponding to the quantized value $q$ when $γ = 0$. In practice, $q$ is represented using a fixed number of bits. For example, in 8bit quantization, $q$ is an 8-bit integer. Note that in general, $q$ may not fit within the number of bits provided, in which case the closest representable value is used.

One of the most important operations when doing forward inference in DNNs is matrix multiplication. Suppose we have three $N × N$ matrices of real numbers, where the third matrix is equal to the product of the first two matrices. Denote the entries of these 3 matrices as $r^{(i,j)}_α$, where $α ∈ \{1, 2, 3\}$ and $0 ≤ i, j ≤ N − 1$. Their quantization parameters are $(s_α, z_α)$ (in general, different quantization parameters may be used for different neurons in a DNN). We use $q^{(i,j)}_α$ to denote the quantized entries of these 3 matrices. Based on the quantization scheme $r^{(i,j)}_α = s_α(q^{(i,j)}_α− z_α)$ and the definition of matrix multiplication, we have
$$
s_{3}(q_{3}^{(i,j)}-z_{3})=\sum_{k=0}^{N-1}s_{1}(q_{1}^{(i,k)}-z_{1})s_{2}(q_{2}^{(k,j)}-z_{2}),
$$
which can be rewritten as
$$
q_{3}^{(i,j)}=z_{3}+\frac{s_{1}s_{2}}{s_{3}}\sum_{k=0}^{N-1}(q_{1}^{(i,k)}-z_{1})(q_{2}^{(k,j)}-z_{2}).
$$
Suppose $\mathbf{y} := \text{ReLU} (\mathbf{W} \mathbf{x} + \mathbf{b})$ is the function describing the transformation performed in a single layer of a DNN. Its quantized version $\mathbf{y}_q := g (\mathbf{x}_q, \mathbf{W}_q, \mathbf{b}_q)$can be described by the series of calculations shown in Equation (4), where Wq, bq, xq and yq are the quantized versions of the weight matrix W, bias vector b, input vector x, and output vector y, respectively. Let zx and zy be the zero points of x and y respectively. As the zero points of the weights corresponding to each output neuron may be different, we use zjw to denote the zero point of the weights corresponding to the jth neuron. Similarly, sjw, sx, and sy are the scales for the weight matrix, input, and output, respectively. The ReLU function in the quantized network can be represented as the maximum of the input and the zero point. The calculation for yq := g(xq, Wq, bq) can then be written as:
$$
\begin{array}{rl}
(\text{i}) & {\,\hat{y}_{0}^{j}:=z_{y}+ \displaystyle \frac{s_{w}^{j}s_{x}}{s_{y}}\sum_{i}(w_{q}^{(i,j)}-z_{w}^{j})(x_{q}^{i}-z_{x})+b_{q}^{j}}\\
(\text{ii}) &{\,\hat{y}_{1}^{j}:=Round(\hat{y}_{0}^{j})}\\
(\text{iii}) &{\,\hat{y}_{2}^{j}:=Clip(\hat{y}_{1}^{j},lb,ub)}\\
(\text{iv}) &{\,y_{q}^{j}:=\operatorname*{max}(\hat{y}_{2}^{j},z_{y})}
\end{array} \tag{4}
$$
where $lb$ and $ub$ are the smallest and largest values, respectively, that can be represented by our quantized integer type, e.g., for an 8-bit unsigned type, [lb, ub]=[0, 255]. The Clip function returns the value within [lb, ub] closest to its input. In Pytorch, weights are usually quantized as signed integers while the inputs and outputs of each layer are quantized as unsigned integers. The quantization parameters (i.e., zero points and scales) are computed offline and determined at the time of quantization. In the inference phase, they are constants. Fig. 1 shows a QNN performing an example computation.

The property utilized in this paper for testing the verification efficiency is robustness. Let f : Dn → Om be a neural network classifier, where, for a given input x ∈ Dn, f (x) = {o0(x), o1(x), ..., om−1(x)} ∈ Om represents the confidence values for m classification labels. In general D and O are sets of real numbers, and for quantized neural networks they are sets of integers corresponding to the quantization type. The prediction of x is given as F (x) = arg max0≤i≤m−1 oi(x), and the label space is denoted as Y. The robustness property can be depicted as: given a test point x∗ with label l∗, a neural network is locally robust at point x∗ with respect to a perturbation radius r if the following formula holds:

$$
\forall x \; ( x \in B _ { \infty } ( x _ { * } , r ) \rightarrow F ( x ) = l _ { * } ) \tag{5}
$$

where $B _ { \infty } ( x _ { * } , r ) = \{ x \mid \| x - x _ { * } \| _ { \infty } \leq r \}$ is the perturbation space around $x_∗$ bounded by an $l_∞$-norm ball of radius $r$. The goal of the verifier is to answer whether Equation (5) holds.

## ILP Modeling

In this section, we introduce an ILP formulation for the QNN robustness verification problem. Compared with previous work on QNN verification (Mistry, Saha, and Biswas 2022; Zhang et al. 2022), the main difference is that our encoding correctly models quantization schemes used in mainstream deep learning frameworks (e.g., PyTorch). In addition, unlike (Mistry, Saha, and Biswas 2022), we avoid using floating point variables, as in our experience, it is easier to solve ILP problems than to solve MILP problems. And in contrast to (Zhang et al. 2022), we avoid piecewise constraints which introduce many redundant variables.

In this paper, we use a symbol with a dot (“·”) to denote a variable in our ILP model corresponding to an input to output from some layer of the DNN, e.g., variable $y ̇$.

We show how to encode each step in calculation (4). For step (i), we use the variable $x ̇$ iq for the $i$-th component of the  input and an auxiliary variable $\hat{y}^j_0$ to denote the result. Note that $\hat{y}^j_0$ is a temporary variable and is not of integer type. The introduction of this symbol is for the sake of convenience, and we show how to eliminate it below.
$$
\hat { y } _ { 0 } ^ { j } = z _ { y } + \frac { s _ { w } ^ { j } s _ { x } } { s _ { y } } \sum _ { i } ( w _ { q } ^ { ( i , j ) } - z _ { w } ^ { j } ) ( \dot { x } _ { q } ^ { i } - z _ { x } ) + b _ { q } ^ { j }
$$


$$
\left\{ \begin{array} { l l } { \dot { \hat { y } } _ { 1 } ^ { j } - \hat { y } _ { 0 } ^ { j } \leq 0 . 5 } \\ { \hat { y } _ { 0 } ^ { j } - \dot { \hat { y } } _ { 1 } ^ { j } \leq 0 . 5 - \varepsilon , } \end{array} \right.
$$

$$

$$

$$
\left\{ \begin{array} { l l } { \dot { \hat { y } } _ { 1 } ^ { j } - z _ { y } - \frac { s _ { w } ^ { j } s _ { x } } { s _ { y } } \sum _ { i } ( w _ { q } ^ { ( i , j ) } - z _ { w } ^ { j } ) ( \dot { x } _ { q } ^ { i } - z _ { x } ) - b _ { q } ^ { j } \leq 0 . 5 } \\ { z _ { y } + \frac { s _ { w } ^ { j } s _ { x } } { s _ { y } } \sum _ { i } ( w _ { q } ^ { ( i , j ) } - z _ { w } ^ { j } ) ( \dot { x } _ { q } ^ { i } - z _ { x } ) + b _ { q } ^ { j } - \dot { \hat { y } } _ { 1 } ^ { j } \leq 0 . 5 - \varepsilon } \end{array} \right.
$$


$$
Encode\_max ( z , x , y ) = \left\{ \begin{array} { l l } { b _ { x } + b _ { y } = 1 } \\ { x - z \leq 0 } \\ { y - z \leq 0 } \\ { x - z + M b _ { y } \geq 0 } \\ { y - z + M b _ { x } \geq 0 } \\ { y - x + M b _ { x } \geq 0 } \end{array} \right.
$$


$$
E n c o d e . m a x ( \dot { y } _ { m a x } ^ { j } , \dot { y } _ { 1 } ^ { j } , l b ) \bigcup E n c o d e . m i n ( \dot { y } _ { 2 } ^ { j } , \dot { y } _ { m a x } ^ { j } , u b )
$$

### Encoding for Typical Fusion Layers

$$
\begin{array} { r } { ( \ddot { \bf { i i i } } \oplus \dot { \bf { i v } } ) \; y _ { q } ^ { j } : = C l i p ( \hat { y } _ { 1 } ^ { j } , l b ^ { \prime } , u b ) , } \end{array}
$$



## Interval Analysis

$$
\varepsilon - 0 . 5 + \hat { y } _ { 0 } ^ { j } \le \hat { y } _ { 1 } ^ { j } \le 0 . 5 + \hat { y } _ { 0 } ^ { j }
$$


$$
\begin{array} { r } { \hat { y } _ { m a x } ^ { j } : = R e L U ( \hat { y } _ { 1 } ^ { j } , l b ) , \hat { y } _ { 2 } ^ { j } : = u b - R e L U ( u b - \hat { y } _ { m a x } ^ { j } ) } \end{array}
$$


## Gradient-based Heuristic Search

## Experiments

## Conclusion

In this work, we propose an efficient verification framework for QNNs that offers different trade-offs between scalability and precision. Our verification tool EQV is the first formal verification tool that precisely captures the quantization scheme used in popular deep learning frameworks. Although we focus on verifying adversarial robustness, our method could be generalized to verify other properties of QNNs. Experimental results show that EQV is more efficient and scalable than previously existing approaches. In future work, it would be interesting to formally analyze the difference or equivalence between the original networks and the quantized neural networks or to formally quantify the precision loss due to the quantization process.