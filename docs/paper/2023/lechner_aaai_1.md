# Quantization-Aware Interval Bound Propagation for Training Certifiably Robust Quantized Neural Networks

## Introduction

Quantized neural networks (QNNs) are neural networks that represent their weights and compute their activations using low-bit integer variables. QNNs significantly improve the latency and computational efficiency of inferencing the network for two reasons. 

- First, the reduced size of the weights and activations allows for a much more efficient use of memory bandwidth and caches. 
- Second, integer arithmetic requires less silicon area and less energy to execute than floating-point operations. 

Consequently, dedicated hardware for running QNNs can be found in GPUs, mobile phones, and autonomous driving computers.

Adversarial attacks are a well-known vulnerability of neural networks that raise concerns about their use in safetycritical applications (Szegedy et al. 2013; Goodfellow, Shlens, and Szegedy 2014). These attacks are norm-bounded input perturbations that make the network misclassify samples, despite the original samples being classified correctly and the perturbations being barely noticeable by humans. For example, most modern image classification networks can be fooled when changing each pixel value of the input image by a few percent. Consequently, researchers have tried to train networks that are provably robust against such attacks. 

The two most common paradigms of training robust networks are **adversarial training** (Madry et al. 2018), and **abstract interpretation-based training** (Mirman, Gehr, and Vechev 2018; Wong and Kolter 2018). 

- Adversarial training and its variations perturb the training samples with gradient descent-based adversarial attacks before feeding them into the network (Madry et al. 2018; Zhang et al. 2019; Wu, Xia, and Wang 2020; Lechner et al. 2021). While this improves the robustness of the trained network empirically, it provides no formal guarantees of the network’s robustness due to the incompleteness of gradient descent-based attacking methods, i.e., gradient descent might not find all attacks. 
- Abstract interpretation-based methods avoid this problem by overapproximating the behavior of the network in a forward pass during training. In particular, instead of directly training the network by computing gradients with respect to concrete samples, these algorithms compute gradients of bounds obtained by propagating abstract domains. While the learning process of abstract interpretation-based training is much less stable than a standard training procedure, it provides formal guarantees about the network’s robustness. The interval bound propagation (IBP) method (Gowal et al. 2019) effectively showed that the learning process with abstract interpretation can be stabilized whe.g.adually increasing the size of the abstract domains throughout the training process.

Previous work has considered adversarial training and IBP for floating-point arithmetic neural networks, however robustness of QNNs has received comparatively much less attention. Since it was demonstrated by (Giacobbe, Henzinger, and Lechner 2020) that neural networks may become vulnerable to adversarial attacks after quantization even if they have been verified to be robust prior to quantization, one must develop specialized training and verification procedures in order to guarantee robustness of QNNs. Previous works have proposed several robustness verification procedures for QNNs (Giacobbe, Henzinger, and Lechner 2020; Baranowski et al. 2020; Henzinger, Lechner, and Žikelić 2021), but none of them consider algorithms for learning certifiably robust QNNs. Furthermore, the existing verification procedures are based on constraint solving and cannot be run on GPU or other accelerating devices, making it much more challenging to use them for verifying large QNNs.

In this work, we present the first abstract interpretation training method for the discrete semantics of QNNs. 

- We achieve this by first defining abstract interval arithmetic semantics that soundly over-approximate the discrete QNN semantics, giving rise to an end-to-end differentiable representation of a QNN abstraction. We then instantiate quantization-aware training techniques within these abstract interval arithmetic semantics in order to obtain a procedure for training certifiably robust QNNs.
- Next, we develop a robustness verification procedure which allows us to formally verify QNNs learned via our IBP-based training procedure. 
  - We prove that our verification procedure is **complete**, meaning that for any input QNN it will either prove its robustness or find a counterexample. 
  - This contrasts the case of abstract interpretation verification procedures for neural networks operating over real arithmetic which are known to be **incomplete** (Mirman, Baader, and Vechev 2021a). 
  - The key advantage of our training and verification procedures for QNNs is that it can make use of GPUs or other accelerator devices. 
  - In contrast, the existing verification methods for QNNs are based on constraint solving so cannot be run on GPU.
- Finally, we perform an experimental evaluation showing that our method outperforms existing state-of-the-art certified $L_∞$-robust QNNs. We also elaborate on the limitations of our training method by highlighting how the low precision of QNNs makes IBP-based training difficult.

We summarize our contribution in three points:

- We introduce the first learning procedure for learning robust QNNs. Our learning procedure is based on quantization-aware training and abstract interpretation method. 
- We develop the first complete robustness verification algorithm for QNNs (i.e., one that always terminates with the correct answer) that runs entirely on GPU or other neural network accelerator devices and make it publicly available. 
- We experimentally demonstrate that our method advances the state-of-the-art on certifying $L_∞$-robustness of QNNs.

## Related Work

?> 列了一堆相关工作，但是没有探讨联系

### Abstract interpretations for neural networks 

Abstract interpretation is a method for overapproximating the semantics of a computer program in order to make its formal analysis feasible (Cousot and Cousot 1977). 

Abstract interpretation executes program semantics over abstract domains instead of concrete program states. The method has been adapted to the robustness certification of neural networks by computing bounds on the outputs of neural networks (Wong and Kolter 2018; Gehr et al. 2018; Tjeng, Xiao, and Tedrake 2019). 

- For instance, 
  - **polyhedra** (Katz et al. 2017; Ehlers 2017; Gehr et al. 2018; Singh et al. 2019; Tjeng, Xiao, and Tedrake 2019), 
  - **intervals** (Gowal et al. 2019) , 
  - **hybrid automata** (Xiang, Tran, and Johnson 2018), 
  - **zonotopes** (Singh et al. 2018), 
  - **convex relaxations** (Dvijotham et al. 2018; Zhang et al. 2020; Wang et al. 2021), and 
  - **polynomials** (Zhang et al. 2018b) 
- have been used as abstract domains in the context of neural network verification. 

Abstract interpretation has been shown to be most effective for verifying neural networks when directly incorporating them into gradient descent-based training algorithms by **optimizing the obtained output bounds** as the loss function (Mirman, Gehr, and Vechev 2018; Wong and Kolter 2018; Gowal et al. 2019; Zhang et al. 2020).

Most of the abstract domains discussed above exploit the piecewise linear structure of neural networks, e.g., linear relaxations such as polytopes and zonotopes. However, linear relaxations are less suited for QNNs due to their piecewiseconstant discrete semantics.

### Verification of quantized neural networks

The earliest work on the verification of QNNs has focused on **binarized neural networks (BNNs)**, i.e., 1-bit QNNs (Hubara et al. 2016). 

- In particular, (Narodytska et al. 2018) and (Cheng et al. 2018) have reduced the problem of BNN verification to boolean satisfiability (SAT) instances. Using modern SAT solvers, the authors were able to verify formal properties of BNNs. 
- (Jia and Rinard 2020) further improve the scalability of BNNs by specifically training networks that can be handled by SAT-solvers more efficiently. 
- (Ami.e. al. 2021) developed a satisfiability modulo theories (SMT) approach for BNN verification. 



- Verification for many bit QNNs was first reported in (Giacobbe, Henzinger, and Lechner 2020) by reducing the QNN verification problem to quantifier-free bit-vector satisfiability modulo theory (QF_BV SMT). 
- The SMT encoding was further improved in (Henzinger, Lechner, and Žikelić 2021) by removing redundancies from the SMT formulation. 
- (Baranowski et al. 2020) introduced fixed-point arithmetic SMT to verify QNNs. 
- The works of (Sena et al. 2021, 2022) have studied SMT-based verification for QNNs as well. 
- Recently, (Mistry, Saha, and Biswas 2022) proposed encoding of the QNN verification problem into a mixed-integer linear programming (MILP) instance. 
- IntRS (Li.e. al. 2021) considers the problem of certifying adversarial robustness of quantized neural networks using randomized smoothing. IntRS is limited to $L_2$-norm bounded attacks and only provides statistical instead of formal guarantees compared to our approach.

### Decision procedures for neural network verification

Early work on the verification of floating-point neural networks has employed off-the-shelf tools and solvers. 

- For instance, (Pulina and Tacchella 2012; Katz et al. 2017; Ehlers 2017) employed SMT-solvers to verify formal properties of neural networks. 
- Similarly, (Tjeng, Xiao, and Tedrake 2019) reduces the verification problem to mixed-integer linear programming instances. 

Procedures better tailored to neural networks are based on branch and bound algorithms (Bunel et al. 2018). In particular, these algorithms combine incomplete verification routines (bound) with divide-and-conquer (branch) methods to tackle the verification problem. The speedup advantage of these methods comes from the fact that the bounding methods can be easily implemented on GPU and other accelerator devices (Serre et al. 2021; Wang et al. 2021).

### Quantization-aware training

There are two main strategies for training QNNs: **post-training quantization** and **quantization-aware training** (Krishnamoorthi 2018). In posttraining quantization, a standard neural network is first trained using floating-point arithmetic, which is then translated to a quantized representation by finding suitable fixed-point format that makes the quantized interpretation as close to the original network as possible. Post-training quantization usually results in a drop in the accuracy of the network with a magnitude that depends on the specific dataset and network architecture.

To avoid a significant reduction in accuracy caused by the quantization in some cases, quantization-aware training (QAT) models the imprecision of the low-bit fixed-point arithmetic already during the training process, i.e., the network can adapt to a quantized computation during training. The rounding operations found in the semantics of QNNs are nondifferentiable computations. Consequently, QNNs cannot be directly trained with stochastic gradient descent. Researchers have come up with several ways of circumventing the problem of non-differentiable rounding. 

- The most common approach is the **straight-through gradient estimator (STE)** (Bengio, Léonard, and Courville 2013; Hubara et al. 2017). In the forward pass of a training step, the STE applies rounding operations to computations involved in the QNN, i.e., the weights, biases, and arithmetic operations. However, in the backward pass, the rounding operations are removed such that the error can backpropagate through the network. 
- The approach of (Gupta et al. 2015) uses stochastic rounding that randomly selects one of the two nearest quantized values for a given floating-point value. 
- Relaxed quantization (Louizos et al. 2019) generalizes stochastic rounding by replacing the probability distribution over the nearest two values with a distribution over all possible quantized values. 
- DoReFA-Net (Zhou et al. 2016) combines the straight-through gradient estimator and stochastic rounding to train QNN with high accuracy. The authors observed that quantizing the first and last layer results in a significant drop in accuracy, and therefore abstain from quantizing these two layers.

Instead of having a fixed pre-defined quantization range, i.e., fixed-point format, more recent QAT schemes allow learning the quantization range. 

- PACT (Choi et al. 2018) treats the maximum representable fixed-point value as a free variable that is learned via stochastic gradient descent using a straight-through gradient estimation. 
- The approach of (Jacob et al. 2018) keeps a moving average of the values ranges during training and adapts the quantization range according to the moving average. 
- LQ-Nets (Zhang et al. 2018a) learn an arbitrary set of quantization levels in the form of a set of coding vectors. While this approach provides a better approximation of the real-valued neural network than fixedpoint-based quantization formats, it also prevents the use of efficient integer arithmetic to run the network. 
- MobileNet (Howard et al. 2019) is a specialized network architecture family for efficient inference on the ImageNet dataset and employs quantization as one technique to achieve this target. 
- HAWQ-V3 (Yao et al. 2021) dynamically assigns the number of bits of each layer to either 4-bit, 8-bit, or 32-bit depending on how numerically sensitive the layer is. 
- EfficientNet-lite (Tan and Le 2019) employs a neural architecture search to automatically find a network architecture that achieves high accuracy on the ImageNet dataset while being fast for inference on a CPU.

## Preliminaries

### Quantized neural networks(QNNs) 

Feedforward neural networks are functions $f_{\theta}:\mathbb{R}^{n}\rightarrow \mathbb{R}^{m}$ that consist of several sequentially composed layers $f_{\theta}=l_{1}\circ\ldots\circ l_{s}$, where layers are parametrized by the vector $\theta$ of neural network parame-ters. Quantization is an interpretation of a neural network $f_{\theta}$ that evaluates the network over a fixed point arithmetic and operates over a restricted set of bitvector inputs(Smith et al.1997), e.g. 4 or 8 bits. Formally, given an admissible input set $\mathcal{I}\subseteq\mathcal{\mathbb{R}}^{n}$, we define an interpretation map
$$
⟦\cdot⟧_{\mathcal{I}}:(\mathbb{R}^{n}\rightarrow \mathbb{R}^{m})\rightarrow(\mathcal{I}\rightarrow \mathbb{R}^{m}),
$$

which maps a neural network to its interpretation operat-ing over the input set $\mathcal{I}$. For instance, if $\mathcal{I}=\mathbb{R}^{n}$ then $⟦f_{\theta}⟧_{\mathbb{R}}$ is the idealized real arithmetic interpretation of $f_{\theta}$, whereas $⟦f⟧_{\text{float}32}$ denotes its floating-point 32-bit imple-mentation(Kahan 1996). Given $k\in N$, the k-bit quantization is then an interpretation map $⟦\cdot⟧_{int-k}$ which uses k-bit fixed-point arithmetic. We say that $⟦f_{\theta}⟧_{int-k}$ is a k-bit quantized neural network(QNN).

The semantics of the QNN $⟦f_{\theta}⟧_{\text{int-}k}$ are defined as follows. Let $[\mathbb{Z}]_{k}=\{0,1\}^{k}$ be the set of all bit-vectors of bit-width k. The QNN $⟦f_{\theta}⟧_{int-k}$ then also consists of sequentially com-posed layers $⟦f_{\theta}⟧_{int-k}=l_{1}\circ\ldots\circ l_{s}$, where now each layer is a function $l_{i}:[\mathbb{Z}]_{k}^{n_{i}}\rightarrow[\mathbb{Z}]_{k}^{n_{i+1}}$ that operates over $k$-bit bitvectors and is defined as follows:

$$
\begin{align*} x_{i}^{\prime}&=\sum_{j=1}^{n_{i}}w_{ij}x_{j}+b_{i},\end{align*} \tag{1}
$$

$$
x^{\prime\prime}_{i}=\mathrm{round}(x^{\prime}_{i}, M_{i})=\lfloor x^{\prime}_{i}\cdot M_{i}\rfloor \tag{2}
$$

$$
y_{i}=\sigma_{i}(\min\{2^{N_{i}}-1, x_{i}^{\prime\prime}\}),\tag{3}
$$

Here, $w_{i, j}\in[\mathbb{Z}]_{k}^{n_{i}}$ and $b_{i}\in[\mathbb{Z}]_{k}^{n_{i}}$ for each $1\leq j\leq n_{i}$ and $1\leq i\leq n_{0}$ denote the weights and biases of f which are also bitvectors of appropriate dimension. Note that it is a task of the training procedure to ensure that trained weights and biases are bitvectors, see below. 

1. In eq. (1), the linear map defined by weights $w_{i, j}$ and biases $b_{i}$ is applied to the input values $x_{j}$. 
2. Then, eq.(2) multiplies the result of eq.(1) by $M_{i}$ and takes the floor of the obtained result. This is done in order to scale the result and round it to the nearest valid fixed-point value, for which one typically uses $M_{i}$ of the form $2^{-k}$ for some integer $k$. 
3. Finally, eq.(3) applies an activation function $\sigma_{i}$ to the result of eq.(2) where the result is first "cut-off" if it exceeds $2^{N_{i}}-1$, i.e., to avoid integer overflows, and then passed to the activation function. 

We restrict ourselves to **monotone** activation functions, which will be necessary for our IBP procedure to be correct. This is still a very general assumption which includes a rich class of activation function, e.g. ReLU, sigmoid or tanh activation functions. Furthermore, similarly to most quantization-aware training procedures our method assumes that it is provided with quantized versions of these activation functions that operate over bit-vectors.

### Adversarial robustness for QNNs 

We now formalize the notion of adversarial robustness for QNN classifiers. Let $⟦f_{\theta}⟧_{int-k}:[\mathbb{Z}]_{k}^{n}\rightarrow[\mathbb{Z}]_{k}^{m}$ be a $k$-bit QNN. It naturally defines a classifier with m classes by assuming that it assigns to an input $x\in[\mathbb{Z}]_{k}^{n}$ a label of the maximal output neuron on input value x, i.e. 
$$
y=\operatorname{class}(x)=\operatorname{argmax}_{1\leq i\leq m}⟦f_{\theta}⟧_{\text{int-} k}(x)[i]
$$
 with $⟦f_{\theta}⟧_{\text{int-} k}(x)[i]$ being the value of the ${i}$-th output neuron on input value $x$. If the maximum is attained at multiple output neurons, we assume that $\mathrm{argmax}$ picks the **smallest** index $1\leq i\leq m$ for which the maximum is attained.

Intuitively, a QNN is adversarially robust at a point $x$ if it assigns the same class to every point in some neighbour-hood of $x$. Formally, given $\epsilon>0$, we say that $⟦f_{\theta}⟧_{\text{int-} k}$ is is $\epsilon$-adversarially robust at point $x$ if

$$
\forall x^{\prime}\in[\mathbb{Z}]_{k}^{n}\cdot\|x-x^{\prime}\|_{\infty}<\epsilon\Rightarrow\operatorname{class}\left(x^{\prime}\right)=\operatorname{class}(x),
$$

where $\|\cdot\|_{\infty}$ denotes the $L_{\infty}$-norm. Then, given a finite dataset $\mathcal{D}=\left\{\left(x_{1}, y_{1}\right),\ldots,\left(x_{|\mathcal{D}|}, y_{|\mathcal{D}|}\right)\right\}$ with $x_{i}\in[\mathbb{Z}]_{k}^{n}$ and $y_{i}\in[\mathbb{Z}]_{k}^{m}$ for each $1\leq i\leq|\mathcal{D}|$, we say that $⟦f_{\theta}⟧_{\text{int}-k}$ is $\epsilon$-adversarially robust with respect to the dataset $\mathcal{D}$ if it is $\epsilon$-adversarially robust at each datapoint in $\mathcal{D}$.

## Quantization-aware Interval Bound Propagation

In the previous section, we presented a quantization-aware training procedure for QNNs with robustness guarantees which was achieved by extending IBP to quantized neural network interpretations. We now show that IBP can also be used towards designing a complete verification procedure for already trained feed-forward QNNs. By completeness, we mean that the procedure is guaranteed to return either that the QNN is robust or to produce an adversarial attack.

## A Complete Decision Procedure for QNN Verification

In the previous section, we presented a quantization-aware training procedure for QNNs with robustness guarantees which was achieved by extending IBP to quantized neural network interpretations. We now show that IBP can also be used towards designing a complete verification procedure for already trained feed-forward QNNs. By completeness, we mean that the procedure is guaranteed to return either that the QNN is robust or to produce an adversarial attack.

There are two important novel aspects of the verification procedure that we present in this section. First, to the best of our knowledge this is the first complete robustness verification procedure for QNNs that is applicable to networks with non-piecewise linear activation functions. Existing constraint solving based methods that reduce verification to SMT-solving are complete but they only support piecewise linear activation functions such as ReLU(Krizhevsky and Hinton 2010). These could in theory be extended to more general activation functions by considering more expressive satisfi-ability modulo theories(Clark and Cesare 2018), however this would lead to inefficient verification procedures and our experimental results in the following section already demon-strate the significant gain in scalability of our IBP-based methods as opposed to SMT-solving based methods for ReLU networks. Second, we note that while our IBP-based verification procedure is complete for QNNs, in general it is known that existing IBP-based verification procedures for real arith-metic neural networks are not complete(Mirman, Baader,and Vechev 2021a). Thus, our result leads to an interesting contrast in IBP-based robustness verification procedures for QNNs and for real arithmetic neural networks.

### Verification procedure 

We now describe our robustness verification procedure for QNNs. Its pseudocode is shown in Algorithm 1. Since verifying $\epsilon$-robustness of a QNN with respect to some finite dataset $\mathcal{D}$ and $\epsilon\,>\,0$ is equivalent to verifying $\epsilon$-robustness of the QNN at each datapoint in D, Algorithm 1 only takes as inputs a QNN $[[f_{\theta}]]_{ int-k}$ that operates over bit-vectors of bit-width k, a single datapoint $x\in[Z]_{k}^{n}$ and a robustness radius $\epsilon>0.$ It then returns either ROBUST if $[[f_{\theta}]]_{ int-k}$ is verified to be $\epsilon$-robust at $x\in[Z]_{k}^{n}$, or VULNERABLE if an adversarial attack $||x^{\prime}-x||_{\infty}<\epsilon$ with $class(x^{\prime})=class(x)$.

![image-20250521170112477](./image/image-20250521170112477.png)

The algorithm proceeds by initializing a stack D of abstract intervals to contain a single element $\{(x-\epsilon\cdot 1,x+\epsilon\cdot 1)\},$where $1\in[Z]_{k}^{n}$ is a unit bit-vector of bit-width k. Intuitively,D contains all abstract intervals that may contain concrete adversarial examples but have not yet been processed by the algorithm. The algorithm then iterates through a loop which in each loop iteration processes the top element of the stack Once the stack is empty and the last loop iteration terminates,Algorithm 1 returns ROBUST.

In each loop iteration, Algorithm 1 pops an abstract inter-val(x,x) from D and processes it as follows. First, it uses IBP for QNNs that we introduced before to propagate(\underline{x},\overline{x})in order to compute an abstract interval $(\underline{y},\overline{y})$ that overap-proximates the set of all possible outputs for a concrete input point in $(\underline{x},\overline{x})$. The algorithm then considers three cases.

- First, if $(\underline{y},\overline{y})$ does not violate Equation(11) which charac-terizes violation of robustness by a propagated abstract inter-val, Algorithm 1 concludes that the abstract interval $(\underline{x},\overline{x})$ does not contain an adversarial example and it proceeds to processing the next element of D. 
- Second, if $(\underline{y},\bar{y})$ violates Equation(11), the algorithm uses projected gradient descent restricted to $(\underline{x},\overline{x})$ to search for an adversarial example and returns VULNERABLE if found. Note that the adversarial attack is generated with respect to the quantization-aware representation of the network, thus ensuring that the input space corresponds to valid quantized inputs. 
- Third, if $(\underline{y},\overline{y})$ violates Equation(11) but the adversarial attack could not be found by projected gradient descent, the algorithm refines the abstract interval(,) by splitting it into two smaller subintervals. This is done by identifying
$$
i^{*}=\text{argmax}_{1\leq i\leq n}(\overline{x}[i]-\underline{x}[i])
$$

and splitting the abstract interval $(\underline{x},\overline{x})$ along the $i^{*}$-th dimen-sion into two abstract subintervals $(\underline{x^{\prime}},\overline{x^{\prime}})$,$(\underline{x^{\prime\prime}},\overline{x^{\prime\prime}})$,which are both added to the stack $D$.

### Correctness, termination and completeness

The following theorem establishes that Algorithm 1 is complete, that it terminates on every input and that it is a complete robustness verification procedure. The proof is provided in the extended version of the paper(Lechner et al. 2022).

**Theorem 2.** If Algorithm 1 returns ROBUST then $[[f_{\theta}]]_{\text{int-k}}$ is $\epsilon$-robust at $x\in [Z]^{n}_{k}$. On the other hand, if Algorithm 1 returns VULNERABLE then there exists an adversarial at-tack $||x^{\prime}-x||_{\infty}<\epsilon$ with $class(x^{\prime})\neq class(x)$. Therefore, Algorithm 1 is correct. Furthermore, Algorithm 1 terminates and is guaranteed to return an output on any input. Since Algorithm 1 is correct and it terminates, we conclude that it is also complete.



## Conclusion

In this paper, we introduced quantization-aware interval bound propagation (QA-IBP), the first method for training certifiably robust QNNs. We also present a theoretical result on the existence and upper bounds on the needed size of a robust QNN for a given dataset of 1-dimensional datapoints. Moreover, based on our interval bound propagation method, we developed the first complete verification algorithm for QNNs that may be run on GPUs. We experimentally showed that our training scheme and verification procedure advance the state-of-the-art on certifying L∞-robustness of QNNs.

Nonetheless, our work serves as a new baseline for future research. Promising directions on how to improve upon QA-IBP and potentially overcome its numerical challenges is to adopt advanced quantization-aware training techniques. For instance, dynamical quantization ranges (Choi et al. 2018; Jacob et al. 2018), mixed-precision layers (Zhou et al. 2016; Yao et al. 2021), and automated architecture search (Tan and Le 2019) have shown promising results for standard training QNNs and might enhance QA-IBP-based training procedures as well. Moreover, further improvements may be feasible by adapting recent advances in IBP-based training methods for non-quantized neural networks (Müller et al. 2022) to our quantized IBP variant.

## References

Amir, G.; Wu, H.; Barrett, C.; and Katz, G.2021. An SMT-based approach for verifying binarized neural networks. In International Conference on Tools and Algorithms for the Construction and Analysis of Systems(TACAS).

Baranowski, M. S.; He, S.; Lechner, M.; Nguyen, T. S.; and Rakamaric, Z. 2020. An SMT Theory of Fixed-Point Arith-metic. In International Joint Conference on Automated Rea-soning(IJCAR).

Bengio, Y.; Léonard, N.; and Courville, A.2013. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv preprint arXiv:1308.3432.

Bubeck, S.; and Sellke, M.2021. A Universal Law of Robust-ness via Isoperimetry. In Conference on Neural Information Processing Systems(NeurIPS).

Bunel, R. R.; Turkaslan, I.; Torr, P.; Kohli, P.; and Mudigonda, P. K.2018. A Unified View of Piecewise Linear Neural Network Verification. In Conference on Neural Information Processing Systems(NeurIPS).

Cheng, C.-H.; Nührenberg, G.; Huang, C.-H.; and Ruess, H. 2018. Verification of Binarized Neural Networks via Inter-Neuron Factoring. In Working Conference on Verified Software: Theories, Tools, and Experiments(VSTTE).

Choi, J.; Wang, Z.; Venkataramani, S.; Chuang, P. I.-J.; Srini-vasan, V.; and Gopalakrishnan, K.2018. PACT: Parame-terized Clipping Activation for Quantized Neural Networks. arXiv preprint arXiv:1805.06085.

Clark, B.; and Cesare, T.2018. Satisfiability Modulo Theo-ries. In Clarke, E. M.; Henzinger, T. A.; Veith, H.; and Bloem, R., eds., Handbook of Model Checking. Springer.

Cousot, P.; and Cousot, R. 1977. Abstract interpretation:a unified lattice model for static analysis of programs by construction or approximation of fixpoints. In Symposium on Principles of Programming Languages(POPL).

Dvijotham, K.; Stanforth, R.; Gowal, S.; Mann, T. A.; and Kohli, P. 2018. A Dual Approach to Scalable Verification of Deep Networks. In Conference on Uncertainty in Artificial Intelligence(UAI).

Ehlers, R. 2017. Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks. In International Sympo-sium on Automated Technology for Verification and Analysis(ATVA).

Gehr, T.; Mirman, M.; Drachsler-Cohen, D.; Tsankov, P.; Chaudhuri, S.; and Vechev, M. T. 2018. AI2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation. In IEEE Symposium on Security and Privacy(S&P).

Giacobbe, M.; Henzinger, T. A.; and Lechner, M. 2020. How Many Bits Does it Take to Quantize Your Neural Network?In International Conference on Tools and Algorithms for the Construction and Analysis of Systems(TACAS).

Goodfellow, I. J.; Shlens, J.; and Szegedy, C.2014. Explain-ing and Harnessing Adversarial Examples. arXiv preprint arXiv:1412.6572.

Gowal, S.; Dvijotham, K. D.; Stanforth, R.; Bunel, R.; Qin, C.; Uesato, J.; Arandjelovic, R.; Mann, T.; and Kohli, P.2019. Scalable Verified Training for Provably Robust Image Clas-sification. In IEEE International Conference on Computer Vision(ICCV).

Gupta, S.; Agrawal, A.; Gopalakrishnan, K.; and Narayanan, P.2015. Deep Learning with Limited Numerical Precision. In International Conference on Machine Learning(ICML).

Henzinger, T. A.; Lechner, M.; and Zikeli, D.2021. Scal-able Verification of Quantized Neural Networks. In AAAI Conference on Artificial Intelligence(AAAI).

Howard, A.; Sandler, M.; Chu, G.; Chen, L.-C.; Chen, B.; Tan, M.; Wang, W.; Zhu, Y.; Pang, R.; Vasudevan, V.; et al.2019. Searching for mobilenetv3. In IEEE Conference on Computer Vision and Pattern Recognition(CVPR).

Hubara, I.; Courbariaux, M.; Soudry, D.; El-Yaniv, R.; and Bengio, Y.2016. Binarized neural networks. In Conference on Neural Information Processing Systems(NeurIPS).

Hubara, I.; Courbariaux, M.; Soudry, D.; El-Yaniv, R.; and Bengio, Y.2017. Quantized Neural Networks: Training Neu-ral Networks with Low Precision Weights and Activations. The Journal of Machine Learning Research(JMLR).

Jacob, B.; Kligys, S.; Chen, B.; Zhu, M.; Tang, M.; Howard, A.; Adam, H.; and Kalenichenko, D. 2018. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. In IEEE Conference on Computer Vision and Pattern Recognition(CVPR).

Jia, K.; and Rinard, M.2020. Efficient exact verification of binarized neural networks. In Conference on Neural Infor-mation Processing Systems(NeurIPS).

Kahan, W. 1996. IEEE Standard 754 for Binary Floating-Point Arithmetic. Lecture Notes on the Status of IEEE.

Katz, G.; Barrett, C.; Dill, D. L.; Julian, K.; and Kochenderfer, M. J. 2017. Reluplex: An Efficient SMT Solver for Verify-ing Deep Neural Networks. In International Conference on Computer Aided Verification(CAV).

Kingma, D. P.; and Ba, J.2015. Adam: A Method for Stochas-tic Optimization. In International Conference on Learning Representations(ICLR).

Krishnamoorthi, R. 2018. Quantizing deep convolutional networks for efficient inference: A whitepaper. arXiv preprint arXiv:1806.08342.

Krizhevsky, A.; and Hinton, G.2010. Convolutional Deep Be-lief Networks on CIFAR-10. University of Toronto Preprint.

Lazarus, C.; and Kochenderfer, M. J. 2022. A mixed integer programming approach for verifying properties of binarized neural networks. arXiv preprint arXiv:2203.07078.

Lechner, M.; Hasani, R. M.; Grosu, R.; Rus, D.; and Hen-zinger, T. A. 2021. Adversarial Training is Not Ready for Robot Learning. In IEEE International Conference on Robotics and Automation(ICRA).

Lechner, M.; Zikeli, D.; Chatterjee, K.; Henzinger, T. A.; and Rus, D. 2022. Quantization-aware Interval Bound Prop-agation for Training Certifiably Robust Quantized Neural Networks. arXiv preprint arXiv:2211.16187.

LeCun, Y.; Bottou, L.; Bengio, Y.; and Haffner, P. 1998. GradientBased Learning Applied to Document Recognition. Proceedings of the IEEE.

Lin, H.; Lou, J.; Xiong, L.; and Shahabi, C.2021. Integer-arithmetic-only Certified Robustness for Quantized Neural Networks. In IEEE Conference on Computer Vision and Pattern Recognition(CVPR).

Loshchilov, I.; and Hutter, F. 2019. Decoupled Weight Decay Regularization. In International Conference on Learning Representations(ICLR).

Louizos, C.; Reisser, M.; Blankevoort, T.; Gavves, E.; and Welling, M. 2019. Relaxed Quantization for Discretized Neural Networks. In International Conference on Learning Representations(ICLR).

Madry, A.; Makelov, A.; Schmidt, L.; Tsipras, D.; and Vladu, A. 2018. Towards Deep Learning Models Resistant to Ad-versarial Attacks. In International Conference on Learning Representations(ICLR).

Mirman, M.; Baader, M.; and Vechev, M. 2021a. The Fun-damental Limits of Interval Arithmetic for Neural Networks. arXiv preprint arXiv:2112.05235.

Mirman, M.; Baader, M.; and Vechev, M. T. 2021b. The Fun-damental Limits of Interval Arithmetic for Neural Networks. CoRR, abs/2112.05235.

Mirman, M.; Gehr, T.; and Vechev, M. 2018. Differentiable Abstract Interpretation for Provably Robust Neural Networks. In International Conference on Machine Learning(ICML).

Mistry, S.; Saha, I.; and Biswas, S.2022. An MILP Encoding for Efficient Verification of Quantized Deep Neural Networks. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 41(11): 4445-4456.

Miller, M. N.; Eckert, F.; Fischer, M.; and Vechev, M.2022. Certified Training: Small Boxes are All You Need. arXiv preprint arXiv:2210.04871.

Narodytska, N.; Kasiviswanathan, S.; Ryzhyk, L.; Sagiv, M.; and Walsh, T. 2018. Verifying Properties of Binarized Deep Neural Networks. In AAAI Conference on Artificial Intelli-gence(AAAI).

Pulina, L.; and Tacchella, A. 2012. Challenging SMT solvers to verify neural networks. AI Communications.

Sena, L.; Song, X.; Alves, E.; Bessa, I.; Manino, E.; Cordeiro, L.; et al. 2021. Verifying Quantized Neural Net-works using SMT-Based Model Checking. arXiv preprint arXiv:2106.05997.

Sena, L. H. C.; et al. 2022. Automated verification and refutation of quantized neural networks. Master's thesis, Universidade Federal do Amazonas.

Serre, F.; Miller, C.; Singh, G.; Püschel, M.; and Vechev, M.2021. Scaling Polyhedral Neural Network Verification on GPUs. In Proc. Machine Learning and Systems(MLSys).

Singh, G.; Gehr, T.; Mirman, M.; Püschel, M.; and Vechev, M.2018. Fast and Effective Robustness Certification. In Confer-ence on Neural Information Processing Systems(NeurIPS).

Singh, G.; Gehr, T.; Püschel, M.; and Vechev, M.2019. An Abstract Domain for Certifying Neural Networks. In Sympo-sium on Principles of Programming Languages(POPL).

Smith, S. W.; et al.1997. The Scientist and Engineer's Guide to Digital Signal Processing. California Technical Pub. San Diego.

Szegedy, C.; Zaremba, W.; Sutskever, I.; Bruna, J.; Erhan, D.; Goodfellow, I.; and Fergus, R. 2013. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199.

Tan, M.; and Le, Q. V.2019. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In International Conference on Machine Learning(ICML).

Tjeng, V.; Xiao, K. Y.; and Tedrake, R. 2019. Evaluating Robustness of Neural Networks with Mixed Integer Program-ming. In International Conference on Learning Representa-tions(ICLR).

Tsipras, D.; Santurkar, S.; Engstrom, L.; Turner, A.; and Madry, A. 2018. Robustness May Be at Odds with Accuracy. In International Conference on Learning Representations(ICLR).

Wang, S.; Zhang, H.; Xu, K.; Lin, X.; Jana, S.; Hsieh, C.-J.; and Kolter, J. Z. 2021. Beta-crown: Efficient bound propa-gation with per-neuron split constraints for neural network robustness verification. In Conference on Neural Information Processing Systems(NeurIPS).

Wong, E.; and Kolter, Z.2018. Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Poly-tope. In International Conference on Machine Learning(ICML).

Wu, D.; Xia, S.-T.; and Wang, Y.2020. Adversarial weight perturbation helps robust generalization. In Conference on Neural Information Processing Systems(NeurIPS).

Xiang, W.; Tran, H.-D.; and Johnson, T. T. 2018. Output Reachable Set Estimation and Verification for Multi-Layer Neural Networks. IEEE Transactions on Neural Networks and Learning Systems.

Xiao, H.; Rasul, K.; and Vollgraf, R.2017. Fashion-MNIST:a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv preprint arXiv:1708.07747.

Yao, Z.; Dong, Z.; Zheng, Z.; Gholami, A.; Yu, J.; Tan, E.; Wang, L.; Huang, Q.; Wang, Y.; Mahoney, M.; et al. 2021. Hawq-v3: Dyadic neural network quantization. In Interna tional Conference on Machine Learning(ICML). PMLR.

Zhang, D.; Yang, J.; Ye, D.; and Hua, G.2018a. LQ-Nets:Learned Quantization for Highly Accurate and Compact Deep Neural Networks. i.e.ropean Conference on Com-puter Vision(ECCV).

Zhang, H.; Chen, H.; Xiao, C.; Gowal, S.; Stanforth, R.; Li, B.; Boning, D. S.; and Hsieh, C.2020. Towards Stable and Efficient Training of Verifiably Robust Neural Networks. In International Conference on Learning Representations(ICLR).

Zhang, H.; Weng, T.-W.; Chen, P.-Y.; Hsieh, C.-J.; and Daniel, L.2018b. Efficient Neural Network Robustness Certification with General Activation Functions. In Conference on Neural Information Processing Systems(NeurIPS).

Zhang, H.; Yu, Y.; Jiao, J.; Xing, E.; e.g.aoui, L.; and Jordan, M. 2019. Theoretically Principled Trade-off between Robustness and Accuracy. In International Conference on Machine Learning(ICML).

Zhou, S.; Wu, Y.; Ni, Z.; Zhou, X.; Wen, H.; and Zou, Y.2016. DoReFa-Net: Training Low Bitwidth Convolutional Neu-ral Networks with Low Bitwidth Gradients. arXiv preprint arXiv:1606.06160.