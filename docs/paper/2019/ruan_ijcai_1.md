# Global Robustness Evaluation of Deep Neural Networks with Provable Guarantees for the $L_0$ Norm

<center>Wenjie Ruan, Min Wu, Youcheng Sun, Xiaowei Huang, Daniel Kroening, and Marta Kwiatkowska</center>

Deployment of deep neural networks (DNNs) in safety- or security-critical systems requires provable guarantees on their correct behaviour. A common requirement is robustness to adversarial perturbations in a neighbourhood around an input. In this paper we focus on the $L_0$ norm and aim to compute, for a trained DNN and an input, the maximal radius of a safe norm ball around the input within which there are no adversarial examples. Then we define global robustness as an expectation of the maximal safe radius over a test data set. We first show that the problem is NP-hard, and then propose an approximate approach to iteratively compute lower and upper bounds on the network’s robustness. The approach is 

- **anytime**, i.e., it returns intermediate bounds and robustness estimates that are gradually, but strictly, improved as the computation proceeds; 
- **tensor-based**, i.e., the computation is conducted over a set of inputs simultaneously, instead of one by one, to enable efficient GPU computation; 
- and has **provable guarantees**, i.e., both the bounds and the robustness estimates can converge to their optimal values. 

Finally, we demonstrate the utility of the proposed approach in practice to compute tight bounds by applying and adapting the anytime algorithm to a set of challenging problems, including global robustness evaluation, competitive $L_0$ attacks, test case generation for DNNs, and local robustness evaluation on large-scale ImageNet DNNs. We release the code of all case studies via GitHub.

## Introduction

## Problem Formulation

**Definition 1 (Safe Norm Ball).** Given a network $f : \mathbb{R}^n \to \mathbb{R}^m$, an input $x_0 \in \mathbb{R}^n$, a distance metric $\| \cdot \|_D$ and a real number $d \in \mathbb{R}$, a norm ball $B(f, x_0, \| \cdot \|_D, d)$ is a subspace of $x_0 \in \mathbb{R}^n$ such that  
$$
B(f, x_0, \| \cdot \|_D, d) = \{ x | \| x_0 - x \|_D \le d \}
$$
The number $d$ is called the **radius** of $B(f, x_0, \| \cdot \|_D, d)$. A norm ball $B(f, x_0, \| \cdot \|_D, d)$ is **safe** if for all $x \in B(f, x_0, \| \cdot \|_D, d)$ we have $cl(f, x) = cl(f, x_0)$.

**Definition 2 (Maximum Radius of a Safe Norm Ball).** Let $d$ be the radius of a safe norm ball $B(f, x_0, \| \cdot \|_D, d)$. If for all $d' > d$ we have that $B(f, x_0, \| \cdot \|_D, d')$ is not safe, then $d$ is called the maximum safe radius, denoted by $d_m(f, x_0,\| \cdot \|_D)$. 

**Definition 3 (Robustness Evaluation).** Given a network $f$, a finite set $\mathcal{T}_0$ of inputs, and a distance metric $\| \cdot \|_D$, the robustness evaluation, denoted as $R(f, \mathcal{T}_0, \| \cdot \|_D)$, is an optimisation problem:
$$
\begin{array}{rl}
& \min_{\mathcal{T}} \| \mathcal{T}_0 - \mathcal{T} \|_D \\
\text{s.t.} & cl(f, x_i) \neq cl(f, x_{0,i}), \text{for } i = 1, \dots, |\mathcal{T}_0|
\end{array}
$$
where $\mathcal{T} = (x_i)_{i=1\dots|\mathcal{T}_0|}$, $\mathcal{T}_0 = (x_{0,i})_{i=1\dots|\mathcal{T}_0|}$, and $x_i, x_{0,i} \in [0, 1]^n$.

## Anytime Robustness Evaluation

**Definition 4 (Sequences of Bounds).** Given a robustness evaluation problem $R(f, \mathcal{T}_0, \| \cdot \|_D)$, a sequence $L(\mathcal{T}_0) = \{l_1, l_2, \dots , l_k\} ∈ R$ is an incremental lower bound sequence if, for all $1 ≤ i < j ≤ k$, we have $l_i ≤ l_j ≤ R(f, \mathcal{T}_0, \| \cdot \|_D)$. The sequence is **strict**, denoted as $L_s(\mathcal{T}_0)$, if for all $1 ≤ i < j ≤ k$, we have either $l_i < l_j$ or $l_i = l_j = R(f, \mathcal{T}_0, \| \cdot \|_D)$. Similarly, we can define a decremental upper bound sequence $U(\mathcal{T}_0)$ and a strict decremental upper bound sequence $U_s(\mathcal{T}_0)$.

**Definition 5 (Anytime Robustness Evaluation).** For a given range $[l_t, u_t]$, we define its centre and radius as follows.  
$$
U_c(l_t, u_t) = \frac{1}{2} (l_t + u_t) \text{ and } U_r(l_t, u_t) = \frac{1}{2} (u_t − l_t).
$$
The anytime evaluation of $R(f, \mathcal{T}_0, \| \cdot \|_D)$ at time $t$, denoted as $R_t(f, \mathcal{T}_0, \| \cdot \|_D)$, is the pair $(U_c(l_t, u_t), U_r(l_t, u_t))$.

## Tensor-based Algorithms for Upper and Lower Bounds

We present our approach to generate the sequences of bounds.

**Definition 6 (Complete Set of Subspaces for an Input).** Given an input $x_0 ∈ [0, 1]^n$ and a set of $t$ dimensions $T ⊆ \{1, ..., n\}$ such that $|T | = t$, the **subspace** for $x_0$, denoted by $X_{x_0,T}$, is a set of inputs $x ∈ [0, 1]^n$ such that $x(i) ∈ [0, 1]$ for $i ∈ T$ and $x(i) = x_0(i)$ for $i ∈ \{1, ..., n\} \setminus T$. Furthermore, given an input $x_0 ∈ [0, 1]^n$ and a number $t ≤ n$, we define 
$$
\mathcal{X} (x_0, t) = \{ X_{x_0,T} | T ⊆ \{1, ..., n\}, |T | = t \}
$$
as the **complete set of subspaces** for input $x_0$.

?> We know that $|X_{x_0,t}| = (\aleph_1)^{t}$ and $\displaystyle |\mathcal{X} (x_0, t)| = {n \choose t}$.



**Definition 7 (Subspace Sensitivity).** Given an input subspace $X ⊆ [0, 1]^n$, an input $x_0 ∈ [0, 1]^n$ and a label $j$, the **subspace sensitivity** w.r.t. $X$, $x_0$, and $j$ is defined as  
$$
S(X, x_0, j) = c_j(x_0) − \inf_{x∈X} c_j(x).
$$
Let $t$ be an integer. We define the subspace sensitivity for $\mathcal{T}_0$ and $t$ as  
$$
S(\mathcal{T}_0, t) = (S(X_{x_0} , x_0, j_{x_0} ))_{X_{x_0} ∈X (x_0,t),x_0∈\mathcal{T}_0}
$$
where $j_{x_0} = \arg \max_{i∈\{1,...,m\}}  c_i(x_0)$ is the classification label of $x_0$ by network $f$.

?> We know that
   $$S(\mathcal{T}_0, t) \in \mathbb{R}^{{n \choose t}, |\mathcal{T}_0|}$$


> 这里为什么只看$j_{x_0}$这个维度？

### Tensor-based Parallelisation for Computing Subspace Sensitivity

**Theorem 1.** Let $T_0$ be a test dataset and $t$ an integer. We have $S(T_0, t) = V(T_0, t) − V(T_0, t)_{\min}$.



### Tensor-based Parallelisation for Computing Lower and Upper Bounds

$$
j = \sum_{k=1, k \neq n}^{N} i_{k} \times \prod_{m=k+1, m \neq n}^{N} I_{m}.
$$


$$
\mathcal{T}(T_0, t) = \text{Tensor}((G(X_{x_i}))_{x_i \in T_0, X_{x_i} \in \mathcal{X}(x_i, t)}) \in \mathbb{R}^{n \times \Delta t \times p \times k}
$$

$$
V(T_0, t)_{\min} = \min(\mathcal{Y}(T_0, t), 1) \in \mathbb{R}^{p \times k}
$$

$$
V(\mathbf{T}_0, t) = {\overbrace{(c_{j_{x_i}}(x_i), \dots, c_{j_{x_i}}(x_i))}^{k}}_{x_i \in \mathbf{T}_0} \in \mathbb{R}^{p \times k}
$$

$$
\mathcal{N}[:, :, i] = \{\mathcal{N}[:, :, i-1] \boxplus \{\mathcal{N}[:, :, i-1] \cap \mathcal{S}(\mathrm{T}_0, t)[:, :, i]\}\} \cup \{\mathcal{S}(\mathrm{T}_0, t)[:, :, i] \boxplus \{\mathcal{N}[:, :, i-1] \cap \mathcal{S}(\mathrm{T}_0, t)[:, :, i]\}\}
$$

$$
T = \{\mathcal{N}_{:,i,m_i} \in \mathbb{R}^{n \times p} | x_i \in T_0\},
$$


$$
\begin{align*}
\min_{\mathbf{L}_1} & \|\mathbf{L}_0 - \mathbf{L}_1\|_D \\
\text{s.t. } & cl(f, z_{0,i} \uplus l_{1,i}) \neq cl(f, z_{0,i} \uplus l_{0,i}) \text{ for } i = 1, \dots, |\mathbf{L}_0|
\end{align*}
$$


### Convergence

We perform convergence analysis of the proposed method. For simplicity, in the proofs we consider the case of a single input $x_0$. The convergence guarantee can be extended easily to a finite set. We first show that grid search can guarantee to find the global optimum given a certain error bound based on the assumption that the neural network satisfies the Lipschitz condition as proved in [19, 31].

**Theorem 2 (Guarantee of the global minimum of grid search).** Assume a neural network $f(x) : [0, 1]^n → \mathbb{R}^m$ is Lipschitz continuous w.r.t. a norm metric $\| \cdot \|_{D}$ and its Lipschitz constant is $K$. By recursively sampling $∆ = 1/\epsilon$ in each dimension, denoted as $\mathcal{X} = \{x_1, ..., x_{∆^n} \}$, the following relation holds:  
$$
\|f_{opt}(x^∗) − \min_{x∈\mathcal{X}} f (x)\|_{D} ≤ K · \| \frac{\epsilon}{2} \mathbf{I}_n\|_{D}
$$
where $f_{opt}(x^∗)$ represents the global minimum value, $\min_{x∈\mathcal{X}} f (x)$ denotes the minimum value returned by grid search, and $\mathbf{I}_n ∈ \mathbb{R}^{n×n}$ is an all-ones matrix.

---

As shown in Sec. 4.2, in each iteration, we apply the grid search to verify the safety of the DNNs (meaning that we preclude adversarial examples) given a lower bound. In combination with Theorem 2, we arrive at the following, which shows the safety guarantee for the lower bounds.

**Theorem 3 (Guarantee for Lower Bounds).** Let $f$ denote a DNN and let $x_0 ∈ [0, 1]^n$ be an input. If our method generates a lower bound $l(f, x_0)$, then $cl(f, x) = cl(f, x_0)$ for all $x$ such that $\|x − x_0\|_0 ≤ l(f, x_0)$. I.e., $f$ is guaranteed to be safe for any pixel perturbations with at most $l(f, x_0)$ pixels.

Theorem 3 (proof in Appendix A.2) shows that the lower bounds generated by our algorithm are the lower bounds of $d_m(f, x_0, \|\cdot\|_D)$. We gradually increase $t = l(f, x_0)$ and re-run the lower bound generation algorithm. Because the number of dimensions of an input is finite, the distance to an adversarial example is also finite. Therefore, the lower bound generation algorithm converges eventually.

---

**Theorem 4 (Guarantee for Upper Bounds).** Let $f$ denote a DNN and $x_0 ∈ [0, 1]^n$ denote an input. Let $u_i(f, x_0)$ be an upper bound generated by our algorithm for any $i > 0$. Then we have $u_{i+1}(f, x_0) ≤ u_i(f, x_0)$ for all $i > 0$, and $\lim_{i→∞} u_i(f, x_0) = d_m(f, x_0, \|\cdot\|_D)$.

The three key ingredients to show that the upper bounds decrease monotonically are: 

- the complete subspaces generated at $t = i$ are always included in the complete subspaces at $t = i + 1$; 
- the pixel perturbation from a subspace with higher priority always results in a larger confidence decrease than those with lower priority; and 
- the tightening strategy is able to exclude the redundant pixel perturbations. 

The details of the proof for Theorem 4 are in Appendix A.3. Finally, we can show that the radius of $[l_i, u_i]$ will converge to 0 deterministically (see Appendix A.4).

## Related Work

### Generation of Adversarial Examples

Existing algorithms compute an upper bound of the maximum safety radius. However, they cannot guarantee to reach the maximum safety radius, while our method is able to produce both lower and upper bounds that provably converge to the maximum safety radius. Most existing algorithms first compute a gradient (either a cost gradient or a forward gradient) and then perturb the input in different ways along the most promising direction on that gradient. 

- FGSM (Fast Gradient Sign Method) [5] is for the $L_∞$ norm. It computes the gradient $∇_X J(θ, x, f(x))$. 
- JSMA (Jacobian Saliency Map based Attack) [18] is for the $L_0$ norm. It calculates the Jacobian matrix of the output of a DNN (in the logit layer) with respect to the input. Then it iteratively modifies one or two pixels until a misclassification occurs. 
- The C&W Attack (Carlini and Wagner) [2] works for the $L_0$, $L_2$ and $L_∞$ norms. It formulates the search for an adversarial example as an **image distance minimisation problem**. The basic idea is to introduce a new optimisation variable to avoid box constraints (image pixels need to lie within $[0,1]$). 
- DeepFool [16] works for the $L_2$ norm. It iteratively linearises the network around the input $x$ and moves across the boundary by a minimal step until reaching a misclassification. 
- VAT (Visual Adversarial Training) [15] defines a KL-divergence at an input based on the model’s robustness to the local perturbation of the input, and then perturbs the input according to this KL-divergence. 

We focus on the $L_0$ norm. We have shown experimentally that for this norm, our approach dominates all existing approaches. We obtain tighter upper bounds at lower computational cost.

### Safety Verification and Reachability Analysis

The approaches aim to not only find an upper bound but also provide guarantees on the obtained bound. There are two ways of achieving safety verification for DNNs. 

- The first is to reduce the problem to a **constraint solving problem**. Notable works include, e.g., [9, 21]. However, they can only work with small networks that have hundreds of hidden neurons. 
- The second is to discretise the vector spaces of the input or hidden layers, and then apply exhaustive search algorithms or Monte-Carlo tree search algorithm on the discretised spaces. The guarantees are achieved by establishing local assumptions such as minimality of manipulations in [8] and minimum confidence gap for Lipschitz networks in [31, 32]. Moreover, 
  - [12] considers determining if an output value of a DNN is reachable from a given input subspace, and reduces the problem to a MILP problem; and 
  - [3] considers the range of output values from a given input subspace. 


Both approaches can only work with small networks. We also mention [19], which computes a lower bound of local robustness for the $L_2$ norm by propagating relations between layers backward from the output. It is incomparable with ours because of the different distance metrics. The bound is loose and cannot be improved (i.e., no convergence). Recently, some researchers use abstract interpretation to verify the correctness of DNNs [4,14]. Its basic idea is to use abstract domains (represented as e.g., boxes, zonotopes, polyhedra) to over-approximate the computation of a set of inputs. In recent work [6] the input vector space is partitioned using clustering and then the method of [9] is used to check the individual partitions. DeepGO [22, 23] shows that most known layers of DNNs are Lipschitz continuous and presents a verification approach based on global optimisation.

However, none of the verification tools above are workable on $L_0$-norm distance in terms of providing the anytime and guaranteed convergence to the true global robustness. Thus, the proposed tool, $L_0$-TRE, is a supplementary to existing research on safety verification of DNNs.

## Conclusions

In this paper, to evaluate global robustness of a DNN over a testing dataset, we present an approach to iteratively generate its lower and upper bounds. We show that the bounds are gradually, and strictly, improved and eventually converge to the optimal value. The method is anytime, tensor-based, and offers provable guarantees. We conduct experiments on a set of challenging problems to validate our approach.

## References

1. Carlini, N., Katz, G., Barrett, C., Dill, D.L.: Ground-truth adversarial examples. arXiv preprint arXiv:1709.10207 (2017)
2. Carlini, N., Wagner, D.: Towards evaluating the robustness of neural networks. In: Security and Privacy (SP), 2017 IEEE Symposium on. pp. 39–57. IEEE (2017)
3. Dutta, S., Jha, S., Sanakaranarayanan, S., Tiwari, A.: Output range analysis for deep neural net- works. arXiv preprint arXiv:1709.09130 (2017)
4. Gehr,T., Mirman,M.,Drachsler-Cohen,D.,Tsankov,P.,Chaudhuri,S.,Vechev,M.T.:AI2:Safety and robustness certification of neural networks with abstract interpretation. In: 2018 IEEE Sym- posium on Security and Privacy (SP) (2018)
5. Goodfellow, I.J., Shlens, J., Szegedy, C.: Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572 (2014)
6. Gopinath,D.,Katz,G.,Pasareanu,C.S.,Barrett,C.:DeepSafe:Adata-drivenapproachforcheck- ing adversarial robustness in neural networks. In: Automated Technology for Verification and Analysis (ATVA). LNCS, vol. 11138, pp. 3–19. Springer (2018)
7. He,K.,Zhang,X.,Ren,S.,Sun,J.:Deepresiduallearningforimagerecognition.In:Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)
8. Huang,X.,Kwiatkowska,M.,Wang,S.,Wu,M.:Safetyverificationofdeepneuralnetworks.In: CAV 2017. pp. 3–29 (2017)
9. Katz, G., Barrett, C., Dill, D., Julian, K., Kochenderfer, M.: Reluplex: An efficient SMT solver for verifying deep neural networks. In: CAV 2017 (2017)
10. Krizhevsky,A.,Sutskever,I.,Hinton,G.E.:ImageNetclassificationwithdeepconvolutionalneu- ral networks. In: Advances in neural information processing systems. pp. 1097–1105 (2012)
11. Kurakin, A., Goodfellow, I., Bengio, S.: Adversarial examples in the physical world. arXiv
preprint arXiv:1607.02533 (2016)
12. Lomuscio, A., Maganti, L.: An approach to reachability analysis for feed-forward relu neural
networks. CoRR abs/1706.07351 (2017), http://arxiv.org/abs/1706.07351
13. Lundberg, S., Lee, S.: A unified approach to interpreting model predictions. CoRR
abs/1705.07874 (2017), http://arxiv.org/abs/1705.07874
14. Mirman,M.,Gehr,T.,Vechev,M.:Differentiableabstractinterpretationforprovablyrobustneural
networks. In: International Conference on Machine Learning. pp. 3575–3583 (2018)
15. Miyato, T., Maeda, S.i., Koyama, M., Nakae, K., Ishii, S.: Distributional smoothing with virtual
adversarial training. arXiv preprint arXiv:1507.00677 (2015)
16. Moosavi-Dezfooli,S.M.,Fawzi,A.,Frossard,P.:DeepFool:asimpleandaccuratemethodtofool
deep neural networks. In: CVPR 2016. pp. 2574–2582 (2016)
17. Olah, C., Satyanarayan, A., Johnson, I., Carter, S., Schubert, L., Ye, K., Mordvintsev, A.:
The building blocks of interpretability. Distill (2018). https://doi.org/10.23915/distill.00010,
https://distill.pub/2018/building-blocks
18. Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z.B., Swami, A.: The limitations of
deep learning in adversarial settings. In: EuroS&P 2016. pp. 372–387. IEEE (2016)
19. Pec, J., Roels, J., Goossens, B., Saeys, Y.: Lower bounds on the robustness to adversarial pertur-
bations. In: NIPS (2017)
20. Pei, K., Cao, Y., Yang, J., Jana, S.: DeepXplore: Automated whitebox testing of deep learning
systems. In: SOSP 2017. pp. 1–18. ACM (2017)
21. Pulina, L., Tacchella, A.: An abstraction-refinement approach to verification of artificial neural
networks. In: CAV 2010. pp. 243–257 (2010)
22. Ruan, W., Huang, X., Kwiatkowska, M.: Reachability analysis of deep neural networks with
provable guarantees. The 27th International Joint Conference on Artificial Intelligence (IJCAI’18)
(2018)
23. Ruan, W., Huang, X., Kwiatkowska, M.: Reachability analysis of deep neural networks with
provable guarantees. arXiv preprint arXiv:1805.02242 (2018)