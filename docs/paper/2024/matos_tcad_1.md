# Counterexample Guided Neural Network  Quantization Refinement

<center>João Batista P. Matos Jr., Eddie B. de Lima Filho, Iury Bessa, Edoardo Manino, Xidan Song, and Lucas C. Cordeiro</center>

Deploying neural networks (NNs) in low-resource domains is challenging because of their high computing, memory, and power requirements. For this reason, NNs are often quantized before deployment, but such an approach degrades their accuracy. Thus, we propose the counterexample-guided neural network quantization refinement (CEG4N) framework, which combines **search-based quantization** and **equivalence checking**. The former minimizes computational requirements, while the latter guarantees that the behavior of an NN does not change after quantization. We evaluate CEG4N on a diverse set of benchmarks, including large and small NNs. Our technique successfully quantizes the networks in the chosen evaluation set, while producing models with up to 163% better accuracy than state-of-the-art techniques.

## INTRODUCTION

NEURAL networks (NNs) are becoming essential in many applications, such as autonomous driving [1], medicine, security, and other safety-critical domains [2]. However, current state-of-the-art NNs often require substantial computing, memory, and power resources, limiting their applicability [3]. As a result, resource-constrained systems may be unable to run complex NNs, leading to high opportunity costs for businesses.

Quantization techniques can help reduce the resource requirements of NNs [3], [4], [5] by decreasing the bit width required to represent their parameters and intermediate computation steps [4]. In general, different quantization strategies can be used. On the one hand, some studies consider only the quantization of NN weights [6], [7], [8]. On the other hand, other studies provide entire NN frameworks in integer precision, including weights, activation functions, and convolutional layers [4], [9].

The goal is compressing an NN to the smallest possible bitwidth. However, doing so may affect the functional behavior of the resulting NN, making it prone to errors and loss of accuracy [5], [9]. For this reason, existing techniques usually monitor the accuracy degradation of a quantized NN (QNN) with statistical measures defined on the training set [5].

## PRELIMINARIES

## RELATED WORK

### Neural Network Quantization

There are many aspects to consider when deciding to deploy a quantization scheme [5]. For instance, if the goal is to reduce an NN’s size, one can consider quantizing only its weights and biases [5], [9], [28]. However, if the goal is to reduce computation and memory requirements, one can consider quantizing weights, biases, and activation functions [9], [28].

Indeed, the quantization of activation functions can reduce computational and memory costs [9], [28]. However, it raises challenges as it usually requires a calibration step using representative data, prior to quantization, to correctly compute quantization ranges [5], [28]. Conservative approaches based on neurons’ transfer functions can also be used [29], but they might lead to poorly quantized regions. In this regard, the technique proposed here aims to reduce an NN’s size as it applies a method that only quantizes its weights and biases.

In fact, several studies have focused specifically on quantizing weights of NNs [6], [8], [30], [31], [32], [33]. 

- For instance, Courbariaux et al. [30] proposed a method called **binarized NNs**, which lowers the storage costs of an NN’s weight parameters by reducing them to binary values. 

- Other studies have also explored **mixed-precision** quantization techniques, which quantize NN weights while retaining activation functions in full precision. 

- Yuan et al. [32] proposed EvoQ, which uses evolutionary search to achieve mixed precision quantization without access to complete training datasets. 

  ?> 为什么要access训练集？这里不限制在post-training quantization吗？

- Zhou et al. [6] proposed the incremental network quantization, which converts a convolutional NN into a lower precision version whose weights can only be either powers of two or zero, considering weight importance to keep high accuracy. 

  ?> powers of two or zero这里有歧义……

- Finally, some studies store quantized weights with floating-point precision, thus facilitating integration for inference and generalization [8], [33].

  ?> generalization是指什么？

### Quantization-Aware Training

Another important aspect to consider is whether to employ a post-training quantization strategy, as we do in the present paper, or allow for some form of weight retraining. The latter may recover some of the performance lost due to the quantization and has been a very active area of research [34], [35], [36]. One of the fundamental problem is expressing the underlying optimization problem in a gradient-friendly form. In this way, the quantization objective can be included in the regular loss function during training [35].

As the size of NNs has grown larger, more recent work attempts to quantize the training gradients too. As an example, Zhou et al. [37] proposed a method called DoReFa-Net for training convolutional NNs with low bitwidth weights, activations, and gradients. During the training process, parameter gradients are quantized to low bitwidth values, allowing faster training and inference using bit convolution kernels. This approach is efficient on various hardware platforms like CPU, FPGA, ASIC, and GPU.

Alternatively, higher rates of compression can be achieved by using different quantization schemes on different regions of the input. For instance, Huang et al. [38] proposed a dynamic quantization strategy that avoids a nonuniform usage of the available computational resources. Their technique is particularly suited to deployment on hardware accelerators.

Unfortunately, quantization-aware training may break the assumptions of the existing NNE verification tools [11], [12], [13], [39]. Thus, in this article, we focus on post-training quantization.

### Verification of Quantized Neural Networks

Giacobbe et al. [40] are the first to formally investigated the impact of quantization on NNs. They explore how quantization affects the NNs’ robustness and formal verification. They propose a bit-precise SMT-solving approach for determining the satisfiability of first-order logic formulas where variables represent fixed-size bit-vectors. Their study shows that there is no simple and direct correlation or pattern between the robustness and the number of bits of a QNN and makes several significant contributions, including revealing nonmonotonicity in QNN robustness, introducing a complete verification method, and highlighting the limitations of existing approaches.

Henzinger et al. [41] proposed an SMT-based verification method for QNNs, using bit-vector specifications. It requires translating an NN and its safety properties into closed quantifier-free formulas over the theory of fixed-size bitvectors. It performs verification only, focusing on robustness, while CEG4N tackles both NN quantization and verification, trying to find a more compact representation that is sound, with NN translation into formulae done by a verifier.

Mistry et al. [42] discussed the formal verification of QNNs implemented using fixed-point arithmetic. The authors propose a novel methodology for encoding the verification problem into a mixed-integer linear programming (MILP) problem, focusing on the bit-precise semantics of QNNs. Their results demonstrate that their MILP-based technique outperforms state-of-the-art bit-vector encodings by a significant margin.

Song et al. [43] proposed QNNVerifier, which performs SMT-based verification of QNNs. Their technique relies on fixed-point operational models for using the C language as an abstract model, which allowed operations to be encoded in their quantized form, explicitly, thus providing compatibility with SMT solvers. Although this study presents some similarities with ours, the main difference lies in the properties being verified. It checks if a QNN is invariant to adversarial inputs, while CEG4N iteratively verifies if an NN is invariant to quantization, with decoupled quantization and verification.

### Neural Network Equivalence

Our CEG4N framework can in principle accommodate multiple equivalence verification techniques. In addition to those, we mention in Section II, all the following are viable alternatives. First, Büning et al. [11] defined the notion of relaxed equivalence because exact equivalence (see Section II-D) is hard to solve. They choose to encode EPs into MILP. First, the input domain is restricted to radii around a point, where equality is more likely. Then, a less strict relation is used. Finally, two NNs R and T are equivalent if the classification result of R is amongst the top-K largest results of T. It was later extended by Teuber et al. [12].

Furthermore, Eleftheriadis et al. [13] proposed an SMT-based NNE checking scheme based on strict $\epsilon$-Equivalence. The key differences between their work and the EC techniques we use are as follows. We also consider Top-Equivalence, and we encode NNs, EP, and equivalence relation either as a C program or Python code along with a structured description. More recently, Zhang et al. [39] introduced QEBVerif, a method that is capable of verifying the equivalence between an NN and its quantized counterpart when both the weights and the activation tensors are quantized. QEBVerif consists of differential RA (DRA) and MILP-based verification. Similar to the work by Eleftheriadis et al. [13], they mainly focus on $\epsilon$-Equivalence, and their technique cannot be easily extended to Top-Equivalence.

## COUNTEREXAMPLE GUIDED NEURAL NETWORK QUANTIZATION REFINEMENT

$$
\begin{array} { r l } {
	\mathrm {Objective:} } & { \mathcal { N } ^ { o } = \underset { n _ { o } ^ { 0 } , \ldots , n _ { o } ^ { H - 1 } } { \arg \operatorname* { m i n } } \ \sum _ { h \in \mathbb { N } _ { h < H } } p ^ { ( h ) } * n ^ { ( h ) } } \\ { \mathrm { s. t. } } & { f ( x ) \! \leq \! f ^ { q } ( x ) \ \ \ \forall \, x \in \mathcal { H } _ { \mathrm { C E } } ^ { o } } \\ & { n ^ { ( h ) } \geq \! \underline { { N } } \ \ \forall \, n ^ { ( h ) } \in \mathcal { N } ^ { o } } \\ & { n ^ { ( h ) } \leq \overline { { N } } \ \ \forall \ n ^ { ( h ) } \in \mathcal { N } ^ { o } . } 
	\end{array}
$$





## EXPERIMENTAL EVALUATION

## CONCLUSION

We have presented a new method for NN quantization, named CEG4N, which is a post-training quantization technique that provides formal guarantees regarding NN equivalence. It relies on a counterexample-guided optimization technique, where an optimization-based quantizer produces compressed NN candidates. A state-of-the-art verifier then checks such candidates to either prove the equivalence between quantized and original NNs or refute it by providing a counterexample. The latter is then fed to the quantizer to guide it in the search for a viable candidate.

In the proposed framework, scalability is tightly related to the underlying verifier, which, in our experiments, took two forms: SMT solver and GPE. SMT solvers are sensitive to NN complexity [29], which leads to large state spaces, while GPE may face exponential growth in the number of associated star sets [12]. Although that may look like an obstacle, both optimization efforts and different quantization strategies have the potential to alleviate these issues. At the same time, it is worth mentioning that our main target was to demonstrate the feasibility of our methodology, which employed the mentioned verifiers as possible solutions, but it is not restricted to them.

In our future work, we will explore other quantization approaches not limited to the search-based ones and different equivalence-verification techniques based on RA [12] and SMT encoding [13]. For instance, [39] provides a new perspective and interpretation of NN behavior centered around error bound verification. This interpretation is different than ours and may have implications and provide new perspectives and conclusions to our quantization problem. Possible future works may provide a more concise formalization of NN equivalence, incorporate QEBVerif approach into CEG4N and provide a comparison work. Combining new quantization and equivalence-verification techniques will help CEG4N achieve better results while providing a more suitable compromise between accuracy and scalability. Furthermore, we will consider quantization approaches that operate entirely on integer arithmetic, which can potentially improve the scalability of the CEG4N’s verification step. The SMT-encoding of the quantization problem will also be considered, with the goal of both comparing its cost with the one presented by our current proposal and devising a unified framework for NN compression and equivalence.