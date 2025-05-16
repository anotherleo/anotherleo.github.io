# POLAR-Express: Efficient and Precise Formal Reachability Analysis of Neural-Network Controlled Systems

<center>Yixuan Wang, Weichao Zhou, Jiameng Fan, Zhilu Wang, Jiajun Li, Xin Chen, Chao Huang, Wenchao Li, and Qi Zhu</center>

Neural networks (NNs) playing the role of controllers have demonstrated impressive empirical performance on challenging control problems. However, the potential adoption of NN controllers in real-life applications has been significantly impeded by the growing concerns over the safety of these NN-controlled systems (NNCSs). 

In this work, we present POLAR-Express, an efficient and precise formal reachability analysis tool for verifying the safety of NNCSs. POLAR-Express uses Taylor model (TM) arithmetic to propagate TMs layer-bylayer across an NN to compute an overapproximation of the NN. It can be applied to analyze any feedforward NNs with continuous activation functions, such as ReLU, Sigmoid, and Tanh activation functions that cover the common benchmarks for NNCS reachability analysis. 

Compared with its earlier prototype POLAR, we develop a novel approach in POLAR-Express to 

- propagate TMs more efficiently and precisely across ReLU activation functions, and 
- provide parallel computation support for TM propagation, thus significantly improving the efficiency and scalability. 

Across the comparison with six other state-of-the-art tools on a diverse set of common benchmarks, POLAR-Express achieves the best verification efficiency and tightness in the reachable set analysis. POLAR-Express is publicly available at https://github.com/ChaoHuang2018/POLAR_Tool.

## INTRODUCTION

Neural networks (NNs) have been successfully used for decision making in a variety of systems such as autonomous vehicles [1], [2], [3], aircraft collision avoidance systems [4], robotics [5], HVAC control [6], [7], and other autonomous cyber–physical systems (CPSs) [8], [9]. NN controllers can be obtained using machine learning techniques such as reinforcement learning [10], [11], imitation learning [12], [13], and transfer learning [14]. However, the usage of NN controllers raises new challenges in verifying the safety of these systems due to the nonlinear and highly parameterized nature of NNs and their closed-loop formations with dynamical systems [15], [16], [17], [18], and adversarial perturbations [19], [20], [21]. In this work, we consider the reachability verification problem of NN-controlled systems (NNCSs).

Uncertainties around the state, such as those inherent in state measurement or localization systems, or scenarios where the system can start from anywhere in an initial space, require the consideration of an initial state set rather than a single initial state for the reachability problem. Specifically, we define the reachability problem for NNCSs as follows.

Definition 1 (Reachability Problem of NNCSs): The reachability problem of an NNCS is to determine whether the system can reach a given goal state set from any state within an initial state set of the system, whereas the bounded-time version of this problem is to determine the reachability within a given bounded-time horizon.

We show Fig. 1 as an example of this set-based closed-loop reachability analysis. It is worth noting that simulation-based testings [22], which sample initial states from the initial state set, cannot provide formal safety guarantees, such as “no system trajectory from the initial state set will lead to an obstacle collision.” In this article, we consider reachability analysis as the class of techniques that aim at tightly overapproximating the set of all reachable states of the system starting from an initial state set.

Reachability analysis of general NNCSs is notoriously hard due to nonlinearities that exist in both the NN controller and the physical plant. The closed-loop coupling of the NN controller with the plant adds another layer of complexity. To obtain a tight overapproximation of the reachable sets, reachability analysis needs to track state dependencies across the closed-loop system and across multiple time steps. While this problem has been well studied in traditional closed-loop systems without NN controllers [23], [24], [25], [26], [27], [28], it is less clear whether it is important to track the state dependency in NNCSs and how to track the dependency efficiently given the complexity of NNs. This article aims to bring clarity to these questions by comparing different approaches for solving the NNCS reachability problems.

Existing reachability analysis techniques for NNCSs typically use reachability analysis methods for dynamical systems as subroutines. For general nonlinear dynamical systems, the problem of exact reachability is undecidable [29]. Thus, methods for reachability analysis of nonlinear dynamical systems aim at computing a tight overapproximation of the reachable sets [25], [26], [27], [30], [31], [32], [33], [34]. On the other hand, there is also rich literature on verifying NNs. Most of these verification techniques boil down to the problem of estimating or overapproximating the output ranges of the network [35], [36], [37], [38], [39], [40]. The existence of these two bodies of work gives rise to a straightforward combination of NN output range analysis with reachability analysis of dynamical systems for solving the NNCS reachability problem. However, early works have shown that this naive combination with a nonsymbolic interval-arithmetic-based [41] output range analysis suffers from large overapproximation errors when computing the reachable sets of the closed-loop system [15], [16]. The primary reason is the lack of consideration of the interactions between the NN controller and the plant dynamics. Recent advances in the field of NN verification feature more sophisticated techniques that can yield tighter output range bounds and track the input–output dependency of an NN via symbolic bound propagation [38], [40], [42]. This opens up the possibility of improvement for the aforementioned combination strategy by substituting the nonsymbolic intervalarithmetic-based technique with these new symbolic bound estimation techniques.

New techniques have also been developed to directly address the verification challenge of NNCSs. Early works mainly use direct end-to-end overapproximation [15], [16], [43] of the neural-network function, i.e., computing a function approximation of the NN with guaranteed error bounds. While this approach can better capture the input–output dependency of an NN compared to output ranges, it suffers from efficiency and scalability problems due to the need to sample from the input space. This approach is superseded by more recent techniques that leverage layer-by-layer propagation in the NN [17], [44], [45], [46]. Layer-by-layer propagation techniques have the advantage of being able to exploit the structure of the NN. They are primarily based on propagating Taylor models (TMs) layer by layer via TM arithmetic to more efficiently obtain a function overapproximation of the NN.

## BACKGROUND

## PROBLEM FORMULATION

## FRAMEWORK OF POLAR-EXPRESS

## BENCHMARK EVALUATIONS

## CONCLUSION AND FUTURE WORK

We present POLAR-Express, a formal reachability verification tool for NNCSs, which uses layer-by-layer propagation of TMs to compute function overapproximations of NN controllers. We provide a comprehensive comparison of POLAR-Express with existing tools and show that POLARExpress achieves state-of-the-art efficiency and tightness in reachable set computations. On the other hand, current techniques still do not scale well to high-dimensional cases. In our experiment, the performance of Verisig 2.0 degrades significantly for the 6-D examples, and POLAR-Express is also less efficient in the QUAD20 example. 

We believe state dimensions, control step sizes, and the number of total control steps are the key factors in scalability. 

- As TMs are parameterized by state variables, **higher state dimensions** will lead to a more tedious polynomial expression in the TMs. 
- Meanwhile, a **large control step** or a **large number of total control steps** can make it more difficult to propagate the state dependencies across the plant dynamics and across multiple control steps. 

We believe that addressing these scalability issues will be the main subject of future work in NNCS reachability analysis.