# Divide and Slide: Layer-Wise Refinement for  Output Range Analysis of Deep Neural Networks

<center>Chao Huang, Jiameng Fan, Xin Chen, Wenchao Li, and Qi Zhu</center>

In this article, we present a layer-wise refinement method for neural network output range analysis. While approaches such as nonlinear programming (NLP) can directly model the high nonlinearity brought by neural networks in output range analysis, they are known to be difficult to solve in general. We propose to use a convex polygonal relaxation (overapproximation) of the activation functions to cope with the nonlinearity. This allows us to encode the relaxed problem into a mixedinteger linear program (MILP), and control the tightness of the relaxation by adjusting the number of segments in the polygon. Starting with a segment number of 1 for each neuron, which coincides with a linear programming (LP) relaxation, our approach selects neurons layer by layer to iteratively refine this relaxation. To tackle the increase of the number of integer variables with tighter refinement, we bridge the propagation-based method and the programming-based method by dividing and sliding the layerwise constraints. Specifically, given a sliding number s, for the neurons in layer l, we only encode the constraints of the layers between l − s and l. We show that our overall framework is sound and provides a valid overapproximation. Experiments on deep neural networks demonstrate significant improvement on output range analysis precision using our approach compared to the state-of-the-art.

## INTRODUCTION

Neural networks have shown promising applications in a variety of domains, including safety-critical systems, such as self-driving cars and medical devices. However, to ensure system safety and security, more formal analysis of neural networks is needed before they can be widely applied in practice. As observed in [1], some of the key correctness problems of neural networks, such as adversarial robustness [2]–[4] and reachability analysis of neural-network controlled systems [5]–[7], can be converted to the analysis of their output range. Thus, addressing the output range analysis problem is vital to provide guarantees for the safety and security of neural networks.

Informally, output range analysis solves the following problem: given a neural network $ f $ and the input range $ \mathcal{X} $, compute the output range of $ f(\mathcal{X}) $. Since a neural network is highly nonlinear due to the large number of parameters and nonlinear activation functions, it is generally difficult to compute the exact range. In most cases, we use an overapproximation $ \bar{\mathcal{Y}} $ such that $ f(\mathcal{X}) \subseteq \bar{\mathcal{Y}} $. Such overapproximation can provide an explicit bound for determining whether the neural network output falls into an unwanted region. Early work shows that basic interval-bound propagation (IBP) can be used to tackle this problem, but often leads to an overly loose estimation due to the loss of dependencies across layers [8].

State-of-the-art methods for output range analysis mainly fall into two categories: 1) symbolic interval propagation (SIP) [9], [10] and 2) constraint programming (CP) [11], [12], where the overapproximation is computed in different manners. The main drawback with SIP is that it can hardly propagate the dependencies for nonlinear operations across layers, and the performance of these propagation-based methods declines with deeper networks. On the other hand, CP-based methods need to solve a large nonlinear programming (NLP) problem encoding the entire network and suffer from the curse of dimension.

In this article, we propose a layer-wise refinement method that bridges propagation-based methods with mixed-integer linear programming (MILP) by using sliding windows. Specifically, we first compute the interval relaxation for each operation with a propagation-based method as the initialization step. Based on the initial range, we use a linear programming (LP) relaxation approach to better approximate the variable range. Then, the relaxation can be further tightened by the MILP encoding. Our approach iteratively improves the approximation precision by increasing the number of integer variables. In addition, we refine the variable range such that fewer integer variables are needed to achieve a similar approximation precision. Furthermore, to alleviate the complexity of MILP brought from the large number of integer variables, we encode the constraints in a propagation manner, which divides and slides the neural network by layers and handles the encoding within each sliding windows. Intuitively, given a length of sliding window $ s $, for the neuron in layer $ l $, we only encode the constraints of the layers between $ l-s $ and $ l $. With these methods, we can effectively manage the size of MILP and handle deep networks.

## RELATED WORK

## PROBLEM FORMULATION

**Notation**: Throughout this article, we use ℝ to denote the set of real numbers, and ℝⁿ to denote the n-dimensional space. Intervals are represented by their endpoints. The set {x ∈ ℝ | a ≤ x ≤ b} is denoted by [a,b]. Arrays can be multidimensional. Given an n-dimensional (n-D) real-valued array M⃗, we use sum(M⃗) to denote the sum of all elements in M⃗, max(M⃗) for the maximum element in M⃗, and M⃗[i₁]⋯[iₙ] to denote the element at position (i₁, ..., iₙ). We represent a section of elements in a dimension from index i to j (i ≤ j) with i:j. Given two n-D arrays M⃗₁ and M⃗₂ of the same size, M⃗₁ ⊙ M⃗₂ denotes their element-wise product. We denote M⃗₁ ∼ M⃗₂ for ∼ ∈ {<, >, ≤, ≥, =} if all corresponding elements satisfy the relation. We use · for scalar multiplication.

**CNN Operations**: We consider convolutional neural networks (CNNs), which process inputs through layers including:

1. **Convolution Operation**: Transforms input image to feature maps using filters, biases, and strides.  
   - Input: X⃗ ∈ ℝ^(W×H×D)  
   - Output: Y⃗ ∈ ℝ^((W-I_F)/S_W +1)×((H-J_F)/S_H +1)  
   - Formula:  
   - 
     
     
     $$
     Y⃗[i][j] = \text{sum}(X⃗[(i-1)S_W:(i-1)S_W + I_F][(j-1)S_H:(j-1)S_H + J_F][1:D] ⊙ W_F) + b_F
     $$
2. **Activation Function**: Applies nonlinearity (ReLU, sigmoid, tanh) element-wise.  
   
   - ReLU: σ(x) = max(0, x)  
   - Sigmoid: σ(x) = 1/(1 + e⁻ˣ)  
   - Tanh: σ(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)  
3. **Pooling Layer**: Extracts dominant features via max or average pooling.  
   - Max Pooling:  
     $$
     Y⃗[i][j][k] = \max(X⃗[(i-1)S_W:(i-1)S_W + I_F][(j-1)S_H:(j-1)S_H + J_F][k])
     $$
   - Average Pooling:  
     $$
     Y⃗[i][j][k] = \frac{1}{I_F J_F} \text{sum}(X⃗[(i-1)S_W:(i-1)S_W + I_F][(j-1)S_H:(j-1)S_H + J_F][k])
     $$
4. **Flatten Layer**: Transforms 3-D array X⃗ ∈ ℝ^(W×H×D) to 1-D array Y⃗:  
   $$
   Y⃗[(k-1)·W·H + (j-1)·W + i] = X⃗[i][j][k]
   $$
5. **Fully Connected (FC) Layer**: Applies affine transformation Y⃗ = W_A X⃗ + b_A followed by activation.

**Network Output**:  
For a CNN with operations OP₁, ..., OPₙ, the output Y⃗ is:  
$$
Y⃗ = OPₙ(OPₙ₋₁(⋯OP₁(X⃗)⋯))
$$

**Output Range Analysis Problem**:  
Given input range 𝒳 (interval array), compute the output range 𝒴 such that:  
$$
𝒴 = \{OPₙ(⋯OP₁(X⃗)⋯) | ∀X⃗, 𝒳 ≤ X⃗ ≤ 𝒳̄\}
$$
where 𝒳 and 𝒳̄ are lower/upper bounds for 𝒳. This requires solving complex nonlinear optimization problems (e.g., for upper bound of Y⃗[1]):  
$$
\max(Y⃗[1]) \text{ s.t. } Y⃗ = σ(W_A \text{Flat}(\text{MaxP}(σ(\text{Conv}(X⃗)))) + b_A) ∧ X⃗ ∈ 𝒳
$$
Due to nonlinearity (activation, max-pooling) and high dimensionality, exact solutions are intractable. Our goal is to compute tight overapproximations via MILP relaxations.

## References

[1] W. Ruan, X. Huang, and M. Kwiatkowska, "**Reachability analysis of deep neural networks with provable guarantees**," in Proc. Int. Joint Conf. Artif. Intell., 2018, pp. 2651-2659.

[2] C. Szegedy et al., "**Intriguing properties of neural networks**," 2013. [Online]. Available: arXiv:1312.6199.

[3] I.J. Goodfellow, J. Shlens, and C. Szegedy, "**Explaining and harnessing adversarial examples**," 2014. [Online]. Available: arXiv:1412.6572.

[4] O. Bastani, Y. Ioannou, L. Lampropoulos, D. Vytinotis, A. Nori, and A. Criminisi, "**Measuring neural net robustness with constraints**," in Proc. Adv. Neural Inf. Process. Syst., 2016, pp. 2613-2621.

[5] C. Huang, J. Fan, W. Li, X. Chen, and Q. Zhu, "**ReachNN: Reachability analysis of neural-network controlled systems**," ACM Trans. Embedded Comput. Syst., vol. 18, no. 5, pp. 1-22, 2019.

[6] S. Dutta, X. Chen, and S. Sankaranarayanan, "**Reachability analysis for neural feedback systems using regressive polynomial rule inference**," in Proc. Hybrid Syst. Comput. Control (HSCC), 2019, pp. 157-168.

[7] R. Ivanov, J. Weimer, R. Alur, G. J. Pappas, and I. Lee, "**Verisig: Verifying safety properties of hybrid systems with neural network controllers**," in Proc. 22nd ACM Int. Conf. Hybrid Syst. Comput. Control, 2019, pp. 169-178.

[8] W. Xiang and T. T. Johnson, "**Reachability analysis and safety verification for neural network control systems**," 2018. [Online]. Available: arXiv:1805.09944.

[9] S. Wang, K. Pei, J. Whitehouse, J. Yang, and S. Jana, "**Formal security analysis of neural networks using symbolic intervals**," in Proc. USENIX Security Symp., 2018, pp. 1599-1614.

[10] G. Singh, T. Gehr, M. Mirman, M. Püschel, and M. Vechev, "**Fast and effective robustness certification**," in Proc. Adv. Neural Inf. Process. Syst., 2018, pp. 10802-10813.

[11] G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer, "**Reuplex: An efficient SMT solver for verifying deep neural networks**," in Proc. Int. Conf. Comput. Aided Verification, 2017, pp. 97-117.

[12] S. Dutta, S. Jha, S. Sankaranarayanan, and A. Tiwari, "**Output range analysis for deep feedforward neural networks**," in Proc. NASA Formal Methods Symp., 2018, pp. 121-138.

[13] T. Gehr, M. Mirman, D. Drachsler-Cohen, P. Tsankov, S. Chaudhuri, and M. Vechev, "**AI2: Safety and robustness certification of neural networks with abstract interpretation**," in Proc. IEEE Symp. Security Privacy (SP), 2018, pp. 3-18.

[14] W. Xiang, H.-D. Tran, and T. T. Johnson, "**Reachable set computation and safety verification for neural networks with RELU activations**," 2017. [Online]. Available: arXiv:1712.08163.

[15] H. Zhang, P. Zhang, and C.-J. Hsieh, "**Recurjac: An efficient recursive algorithm for bounding jacobian matrix of neural networks and its applications**," in Proc. AAAI Conf. Artif. Intell. (AAAI), Dec. 2019, pp. 5757-5764.

[16] H.-D. Tran, S. Bak, W. Xiang, and T.T. Johnson, "**Verification of deep convolutional neural networks using imagestars**," in Proc. Int. Conf. Comput. Aided Verification, 2020, pp. 18-42.

[17] H.-D. Tran et al., "**NNV: The neural network verification tool for deep neural networks and learning-enabled cyber-physical systems**," in Proc. 32nd Int. Conf. Comput.-Aided Verification (CAV), Jul. 2020, pp. 3-17.

[18] G. Singh, T. Gehr, M. Pischel, and M. Vechev, "**Boosting robustness certification of neural networks**," in Proc. Int. Conf. Learn. Represent. (ICLR), 2019, p. 6.

[19] X. Huang, M. Kwiatkowska, S. Wang, and M. Wu, "**Safety verification of deep neural networks**," in Proc. Int. Conf. Comput.-Aided Verification, 2017, pp. 3-29.

[20] C.-H. Cheng, G. Nhrenberg, and H. Ruess, "**Maximum resilience of artificial neural networks**," in Proc. Int. Symp. Autom. Technol. Verification Anal., 2017, pp. 251-268.

[21] M. Fischetti and J. Jo, "**Deep neural networks as 0-1 mixed integer linear programs: A feasibility study**," 2017. [Online]. Available: arXiv:1712.06174.

[22] A. Lomuscio and L. Maganti, "**An approach to reachability analysis for feed-forward RELU neural networks**," 2017. [Online]. Available: arXiv:1706.07351.

[23] V. Tjeng, K. Xiao, and R. Tedrake, "**Evaluating robustness of neural networks with mixed integer programming**," in Proc. Int. Conf. Learn. Represent., 2019, p. 6.

[24] R. Ehlers, "**Formal verification of piece-wise linear feed-forward neural networks**," in Proc. Int. Symp. Autom. Technol. Verification Anal., 2017, pp. 269-286.

[25] E. Wong and Z. Kolter, "**Provable defenses against adversarial examples via the convex outer adversarial polytope**," in Proc. Int. Conf. Mach. Learn., 2018, pp. 5286-5295.

[26] K. Dvijotham, R. Stanforth, S. Gowal, T. A. Mann, and P. Kohli, "**A dual approach to scalable verification of deep networks**," in Proc. UAI, vol. 1, 2018, p. 2.

[27] R. Bunel et al., "**Lagrangian decomposition for neural network verification**," 2020. [Online]. Available: arXiv:2002.10410.

[28] A. Raghunathan, J. Steinhardt, and P.S. Liang, "**Semidefinite relaxations for certifying robustness to adversarial examples**," in Proc. Adv. Neural Inf. Process. Syst., 2018, pp. 10877-10887.

[29] M. Fazlyab, M. Morari, and G.J. Pappas, "**Safety verification and robustness analysis of neural networks via quadratic constraints and semidefinite programming**," 2019. [Online]. Available: arXiv:1903.01287.

[30] P. Prabhakar and Z. R. Afzal, "**Abstraction based output range analysis for neural networks**," in Proc. Adv. Neural Inf. Process. Syst., 2019, pp. 15762-15772.

[31] J. Fan, C. Huang, W. Li, X. Chen, and Q. Zhu, "**ReachNN\*: A tool for reachability analysis of neural-network controlled systems**," in Proc. Int. Symp. Autom. Technol. Verification Anal. (ATVA), 2020.

[32] J. Fan, C. Huang, W. Li, X. Chen, and Q. Zhu, "**Towards verification-aware knowledge distillation for neural-network controlled systems**," in Proc. IEEE/ACM Int. Conf. Comput.-Aided Design (ICCAD), 2019, pp. 1-8.

[33] S. Lawrence, C. L. Giles, A. C. Tsoi, and A. D. Back, "**Face recognition: A convolutional neural-network approach**," IEEE Trans. Neural Netw., vol. 8, no. 1, pp. 98-113, Jan. 1997.

[34] M. Balunovic, M. Baader, G. Singh, T. Gehr, and M. Vechev, "**Certifying geometric robustness of neural networks**," in Proc. Adv. Neural Inf. Process. Syst., 2019, pp. 15287-15297.

[35] X. Huang and L. Zhang, "**Analyzing deep neural networks with symbolic propagation: Towards higher precision and faster verification**," in Proc. Stat. Anal. 26th Int. Symp. (SAS), vol. 11822. Porto, Portugal, Oct. 2019, p. 296.

[36] M. Balunovic and M. Vechev, "**Adversarial training and provable defenses: Bridging the gap**," in Proc. Int. Conf. Learn. Represent., 2020.

[37] S. Boyd, S. P. Boyd, and L. Vandenberghe, **Convex Optimization**. Cambridge, U.K.: Cambridge Univ. Press, 2004.

[38] Gurobi Optimization. (2020). **Gurobi Optimizer Reference Manual**. [Online]. Available: http://www.gurobi.com