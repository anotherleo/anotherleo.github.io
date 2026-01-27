# ReachNN: Reachability Analysis of Neural-Network Controlled Systems

CHAO HUANG, Northwestern University<br/>JIAMENG FAN and WENCHAO LI, Boston University<br/>XIN CHEN, University of Dayton<br/>QI ZHU, Northwestern University

Applying neural networks as controllers in dynamical systems has shown great promises. However, it is critical yet challenging to verify the safety of such control systems with neural-network controllers in the loop. Previous methods for verifying neural network controlled systems are limited to a few specific activation functions. In this work, we propose a new reachability analysis approach based on **Bernstein polynomials** that can verify neural-network controlled systems with a more general form of activation functions, i.e., as long as they ensure that the neural networks are **Lipschitz continuous**. Specifically, we consider abstracting feedforward neural networks with Bernstein polynomials for a small subset of inputs. To quantify the error introduced by abstraction, we provide both theoretical error bound estimation based on the theory of Bernstein polynomials and more practical **sampling based error bound estimation**, following a tight Lipschitz constant estimation approach based on forward reachability analysis. Compared with previous methods, our approach addresses a much broader set of neural networks, including heterogeneous neural networks that contain multiple types of activation functions. Experiment results on a variety of benchmarks show the effectiveness of our approach.

## INTRODUCTION

Data-driven control systems, especially neural-network-based controllers [27, 32, 33], have recently become the subject of intense research and demonstrated great promises. Formally verifying the safety of these systems however still remains an open problem. A Neural-Network Controlled System (NNCS) is essentially a continuous system controlled by a neural network, which produces control inputs at the beginning of each control step based on the current values of the state variables and feeds them back to the continuous system. Reachability of continuous or hybrid dynamical systems with traditional controllers has been extensively studied in the last decades. It has been proven that reachability of most nonlinear systems is undecidable [2, 20]. Recent approaches mainly focus on the **overapproximation of reachable sets** [13, 17, 22, 29, 34, 40]. The main difficulty impeding the direct application of these approaches to NNCS is the hardness of formally characterizing or abstracting the input-output mapping of a neural network.

Some recent approaches considered the problem of computing the output range of a neural network. Given a neural network along with a set of the inputs, these methods seek to compute an interval or a box (vector of intervals) that contains the set of corresponding outputs. These techniques are partly motivated by the study of robustness [16] of neural networks to adversarial examples [38]. Katz et al. [25] propose an SMT-based approach called Reluplex by extending the simplex algorithm to handle ReLU constraints. Huang et al. [23] use a refinement-by-layer technique to prove the absence or show the presence of adversarial examples around the neighborhood of a specific input. General neural networks with Lipschitz continuity are then considered by Ruan et al. [36], where the authors show that a large number of neural networks are Lipschitz continuous and the Lipschitz constant can help in estimating the output range which requires solving a global optimization problem. Dutta et al. [16] propose an efficient approach using mixed integer linear programming to compute the exact interval range of a neural network with only ReLU activation functions.

However, these existing methods cannot be directly used to analyze the reachability of dynamical systems controlled by neural networks. As the behavior of these systems is based on the interaction between the continuous dynamics and the neural-network controller, we need to not only compute the output range but also describe the input-output mapping for the controller. More precisely, we need to compute a tractable function model whose domain is the input set of the controller and its output range contains the set of the controller’s outputs. We call such a function model a higher-order set, to highlight the distinction from intervals which are 0-order sets. Computing a tractable function model from the original model can also be viewed as a form of knowledge distillation [21] from the verification perspective, as the function model should be able to produce comparable results or replicate the outputs of the target neural network on specific inputs.

## PROBLEM STATEMENT

In this section, we describe the reachability of NNCS and a solution framework that computes overapproximations for reachable sets. In the paper, a set of ordered variables $x_1, x_2, . . . , x_m$ is collectively denoted by $x$. For a vector $x$, we denote its $i$-th component by $x_i$.

A NNCS is illustrated in the Figure 1. The plant is the formal model of a physical system or process, defined by an ODE in the form of $\dot{x} = f (x, u)$ such that $x$ are the $n$ state variables and $u$ are the $m$ control inputs. We require that the function $f : \mathbb{R}^m × \mathbb{R}^n → \mathbb{R}^m$ is Lipschitz continuous in $x$ and continuous in $u$, in order to guarantee the existence of a unique solution of the ODE from a single initial state (see [31]).

The controller in our system is implemented as a feed-forward neural network, which can be defined as a function $κ$ that maps the values of $x$ to the control inputs $u$. It consists of $S$ layers, where the first $S − 1$ layers are referred as “hidden layers” and the $S$-th layer represents the network’s output. Specifically, we have
$$
\kappa(x) = \kappa_S(\kappa_{S-1}(\dots \kappa_1(x; W_1, b_1); W_2, b_2); W_S, b_S)
$$

where $W_s$ and $b_s$ for $s = 1, 2, . . . , S$ are learnable parameters as linear transformations connecting two consecutive layers, which is then followed by an element-wise nonlinear activation function. $κ_i (z_{s−1};W_{s−1}, b_{s−1})$ is the function mapping from the output of layer $s − 1$ to the output layer s such that $z_{s−1}$ is the output of layer $s − 1$. An illustration of a neural network is given in Figure 1.

![image-20250512152008632](./image/image-20250512152008632.png ':size=70%')

A NNCS works in the following way. Given a control time stepsize $δ_c > 0$, at the time $t = i{δ_c}$ for $i = 0, 1, 2, \dots$ , the neural network takes the current state $x(i{δ_c})$ as input, computes the input values $u(i{δ_c})$ for the next time step and feeds it back to the plant. More precisely, the plant ODE becomes $\dot{x} = f (x, u(i{δ_c}))$ in the time period of $[iδ_c , (i + 1)δ_c ]$ for i = 0, 1, 2, . . . . Notice that the controller does not change the state of the system but the dynamics. The formal definition of a NNCS is given as below.

**Definition 2.1 (Neural-Network Controlled System).** A neural-network controlled system (NNCS) can be denoted by a tuple $(\mathcal{X}, \mathcal{U}, F , κ, δ_c , X_0)$, where $\mathcal{X}$ denotes the state space whose dimension is the number of state variables, $\mathcal{U}$ denotes the control input set whose dimension is the number of control inputs, $F$ defines the continuous dynamics $\dot{x} = f (x, u)$, $κ : \mathcal{X} → \mathcal{U}$ defines the input/output mapping of the neural-network controller, $δ_c$ is the control stepsize, and $X_0 ⊆ \mathcal{X}$ denotes the initial state set.

Notice that a NNCS is deterministic when the continuous dynamics function $f$ is Lipschitz continuous. The behavior of a NNCS can be defined by its flowmap. The flowmap of a system $(\mathcal{X}, \mathcal{U}, F , κ, δ_c , X_0)$ is a function $φ : X_0 × \mathbb{R}_{≥0} → \mathcal{X}$ that maps an initial state $x_0$ to the state $φ(x_0, t)$, which is the system state at the time $t$ from the initial state $x_0$. Given an initial state $x_0$, the flowmap has the following properties for all $i = 0, 1, 2, . . .$ :

- $φ$ is the solution of the ODE $\dot{x} = f (x, u(i{δ_c}))$ with the initial condition $x(0) = φ (x_0, i{δ_c} )$ in the time interval $t \in [iδ_c , i{δ_c} + δ_c]$; 
- $u (iδ_c ) = κ (φ (x_0, iδ_c ))$.

We call a state $x$ reachable at time $t ≥ 0$ on a system $(\mathcal{X}, \mathcal{U}, F , κ, δ_c , X_0)$, if and only if there is some $x_0 ∈ X_0$ such that $x = φ(x_0, t)$. Then, the set of all reachable states is called the reachable set of the system.

**Definition 2.2 (Reachability Problem).** The reachability problem on a NNCS is to decide whether a given state is reachable or not at time $t ≥ 0$.

In the paper, we focus on the problem of computing the **reachable set** for a NNCS. Since NNCSs are at least as expressive as nonlinear continuous systems, the reachability problem on NNCSs is **undecidable**. Although there are numerous existing techniques for analyzing the reachability of linear and nonlinear hybrid systems [1, 9, 14, 18, 26], none of them can be directly applied to NNCS, since equivalent transformation from NNCS to a **hybrid automaton** is usually very costly due to the large number of locations in the resulting automaton. Even an on-the-fly transformation may lead to a large hybrid automaton in general. Hence, we compute **flowpipe overapproximations** (or flowpipes) for the reachable sets of NNCS.

Similar to the flowpipe construction techniques for the reachability analysis of hybrid systems, we also seek to compute overapproximations for the reachable segments of NNCS. A continuous dynamics can be handled by the existing tools such as SpaceEx [18] when it is linear, and Flow\* [10] or CORA [1] when it is nonlinear. The challenge here is to compute an accurate overapproximation for input/output mapping of the neural-network controller in each control step, and we will do it in the following way.

Given a bounded input interval $X_I$, we compute a model $(g(x), ε)$ where $ε ≥ 0$ such that for any $x ∈ X_I$ , the control input $κ(x)$ belongs the set $\{g(x ) + z | z ∈ B_ε \}$. $g$ is a function of $x$, and $B_ε$ denotes the box $[−ε, ε]$ in each dimension. We provide a summary of the existing works which are close to ours.

**Interval overapproximation.** The methods described in [36, 39] compute intervals as neural-network input/output relation and directly feed these intervals in the reachability analysis. Although they can be applied to more general neural-network controllers, using interval overapproximation in the reachability analysis cannot capture the dependencies of state variables for each control step, and it is reported in [16].

**Exact neural network model.** The approach presented in [24] equivalently transforms the neural-network controller to a hybrid system, then the whole NNCS becomes a hybrid system and the existing analysis methods can be applied. The main limitations of the approach are: (a) the transformation could generate a model whose size is prohibitively large, and (b) it only works on neural networks with sigmoid and tanh activation functions.

**Polynomial approximation with error bound.** In [15], the authors describe a method to produce higher-order sets for neural-network outputs. It is the closest work to ours. In their paper, the approximation model is a piecewise polynomial over the state variables, and the error bound can be well limited when the degrees or pieces of the polynomials are sufficiently high. The main limitation of the method is that it only applies to the neural networks with ReLU activation functions.

## OUR APPROACH

$$
\kappa (x ) \in P (x ) + [-\bar{\epsilon}, \bar{\epsilon}] \text{ for all } x \in X_i
$$

### Bernstein Polynomials for Approximation

$$
B_{f,d} (\mathbf{x}) = \sum_{0 \le k_j \le d_j, j \in \{1, \dots, m\}} f\left(\frac{k_1}{d_1}, \dots, \frac{k_m}{d_m}\right) \prod_{j=1}^m \binom{d_j}{k_j} x_j^{k_j} (1 - x_j)^{d_j - k_j}
$$



### Approximation Error Estimation

$$
\left\|B_{f, d}(x) - f(x)\right\| \leq \frac{L}{2}\left(\sum_{j=1}^{m}\left(1 / d_{j}\right)\right)^{\frac{1}{2}}, \quad \forall x \in I.
$$


$$
\bar{\varepsilon}_t = \frac{L}{2} \left(\sum_{j=1}^m \frac{1}{d_j}\right)^{1/2} \max_{j \in \{1, \ldots, m\}} \{u_j - l_j\}.
$$

$$
\kappa(x) \in \left\{ u \mid u = P_{\kappa,d}(x) + \varepsilon, \varepsilon \in [-\bar{\varepsilon}_t, \bar{\varepsilon}_t] \right\}, \forall x \in X.
$$

$$
\|\partial x / \partial x'\| = \max_{j \in \{1, \dots, m\}} \{u_j - l_j\}
$$


$$
L_{\kappa'}(x') = L_{\kappa(x)} \cdot L_{x(x')} = L \max_{j \in \{1, \dots, m\}} \{u_j - l_j\}.
$$

$$
\|P_{\kappa, d}(x)-\kappa(x)\| = \|B_{\kappa', d}(x')-\kappa'(x')\| \leq \frac{L_{\kappa'}(x')}{2} \left(\sum_{j=1}^m \frac{1}{d_j}\right)^{\frac{1}{2}}.
$$

$$
B_k = \left[ l_1 + \frac{k_1}{p_1}(u_1 - l_1), l_1 + \frac{k_1 + 1}{p_1}(u_1 - l_1) \right] \times \cdots \times \left[ l_m + \frac{k_m}{p_m}(u_m - l_m), l_m + \frac{k_m + 1}{p_m}(u_m - l_m) \right].
$$

$$
\kappa(x) \in \left\{ u \mid u = P_{\kappa,d}(x) + \varepsilon, \varepsilon \in \left[ - \max_{0 \leq k \leq p-1} \bar{\varepsilon}_k, \max_{0 \leq k \leq p-1} \bar{\varepsilon}_k \right] \right\}.
$$

$$
c_k = \left(l_1 + \frac{2k_1 + 1}{2p_1}(u_1 - l_1), \ldots, l_m + \frac{2k_m + 1}{2p_m}(u_m - l_m)\right)
$$



$$
\| P_{\kappa, d}(x) - \kappa(x) \| \leq L \sqrt{\sum_{j=1}^{m} \left(\frac{u_j - l_j}{p_j}\right)^2} + \| P_{\kappa, d}(c_k) - \kappa(c_k) \| .
$$

$$
\begin{align*}
\|P_{\kappa, d}(x) - \kappa(x)\| &\le \|P_{\kappa, d}(x) - P_{\kappa, d}(c_k)\| + \|P_{\kappa, d}(c_k) - \kappa(c_k)\| + \|\kappa(c_k) - \kappa(x)\| \\
&\le L \max_{x \in B_k} \|x - c\| + \|P_{\kappa, d}(c_k) - \kappa(c_k)\| + L \max_{x \in B_k} \|x - c_k\| \\
&= L \sqrt{\sum_{j=1}^m \left(\frac{u_j - l_j}{p_j}\right)^2} + \|P_{\kappa, d}(c_k) - \kappa(c_k)\|.
\end{align*}
$$

$$
\bar{\varepsilon}_s(p) = L \sqrt{\sum_{j=1}^m \left(\frac{u_j - l_j}{p_j}\right)^2} + \max_{0 \le k \le p-1} \|P_{\kappa, d}(c_k) - \kappa(c_k)\| .
$$

$$
\kappa(x) \in \{ u \ | \ u = P_{\kappa,d}(x) + \varepsilon, \varepsilon \in [-\bar{\varepsilon}_s(p), \bar{\varepsilon}_s(p)] \}, \forall x \in X.
$$

$$
\lim_{p \to \infty} \bar{\varepsilon}_s(p) \to \bar{\varepsilon}_{best}.
$$


$$
|\bar{\varepsilon}_s(p) - \bar{\varepsilon}_{best}|
\leq 
\left|\bar{\varepsilon}_s(p) - \max_{0 \leq k \leq p-1} \|P_{\kappa, d}(c_k) - \kappa(c_k)\| \right| = \delta(p) 
$$


$$
p_j = \left\lceil L(u_j - l_j)\sqrt{m/\bar{\delta}} \right\rceil, \quad j = 1, \dots, m.
$$

## EXPERIMENTS

$$
\dot{x}_1 = x_2, \dot{x}_2 = ux_2^2 - x_1,
$$

## DISCUSSION AND OPEN CHALLENGES

## CONCLUSION

In this paper, we address the reachability analysis of neural-network controlled systems, and present a novel approach ReachNN. Given an input space and a degree bound, our approach constructs a polynomial approximation for a neural-network controller based on Bernstein polynomials and provides two techniques to estimate the approximation error bound. Then, leveraging the off-the-shelf tool Flow*, our approach can iteratively compute flowpipes as over-approximate reachable sets of the neural-network controlled system. The experiment results show that our approach can effectively address various neural-network controlled systems. Our future work includes further tightening the approximation error bound estimation and better addressing high-dimensional cases.

## REFERENCES

[1] M. Althoff. 2015. An introduction to CORA 2015. In Proc. of ARCH’15 (EPiC Series in Computer Science), Vol. 34. EasyChair, 120–151.

[2] R. Alur, C. Courcoubetis, N. Halbwachs, T. A. Henzinger, P.-H. Ho, X. Nicollin, A. Olivero, J. Sifakis, and S. Yovine. 1995. The algorithmic analysis of hybrid systems. Theor. Comput. Sci. 138, 1 (1995), 3–34.

[3] Jimmy Ba and Rich Caruana. 2014. Do deep nets really need to be deep?. In Advances in Neural Information Processing Systems. 2654–2662.

[4] Randall D. Beer, Hillel J. Chiel, and Leon S. Sterling. 1989. Heterogeneous neural networks for adaptive behavior in dynamic environments. In Advances in Neural Information Processing Systems. 577–585.

[5] M. Berz and K. Makino. 1998. Verified integration of ODEs and flows using differential algebraic methods on high-order taylor models. Reliable Computing 4 (1998), 361–369. Issue 4.

[6] B. M. Brown, D. Elliott, and D. F. Paget. 1987. Lipschitz constants for the Bernstein polynomials of a Lipschitz continuous function. Journal of Approximation Theory 49, 2 (1987), 196–199.

[7] Cristian Bucilă, Rich Caruana, and Alexandru Niculescu-Mizil. 2006. Model compression. In Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 535–541.

[8] X. Chen. 2015. Reachability Analysis of Non-Linear Hybrid Systems Using Taylor Models. Ph.D. Dissertation. RWTH Aachen University.

[9] X. Chen, E. Ábrahám, and S. Sankaranarayanan. 2012. Taylor model flowpipe construction for non-linear hybrid systems. In Proc. of RTSS’12. IEEE Computer Society, 183–192.

[10] X. Chen, E. Ábrahám, and S. Sankaranarayanan. 2013. Flow*: An analyzer for non-linear hybrid systems. In Proc. of CAV’13 (LNCS), Vol. 8044. Springer, 258–263.

[11] X. Chen and S. Sankaranarayanan. 2016. Decomposed reachability analysis for nonlinear systems. In 2016 IEEE Real-Time Systems Symposium (RTSS). IEEE Press, 13–24.

[12] Louis De Branges. 1959. The stone-weierstrass theorem. Proc. Amer. Math. Soc. 10, 5 (1959), 822–824.

[13] T. Dreossi, T. Dang, and C. Piazza. 2016. Parallelotope bundles for polynomial reachability. In HSCC. ACM, 297–306.

[14] P. S. Duggirala, S. Mitra, M. Viswanathan, and M. Potok. 2015. C2E2: A verification tool for stateflow models. In Proc. of TACAS’15 (LNCS), Vol. 9035. Springer, 68–82.

[15] S. Dutta, X. Chen, and S. Sankaranarayanan. 2019. Reachability analysis for neural feedback systems using regressive polynomial rule inference. In Hybrid Systems: Computation and Control (HSCC). ACM Press, 157–168.

[16] S. Dutta, S. Jha, S. Sankaranarayanan, and A. Tiwari. 2018. Output range analysis for deep feedforward neural networks. In NASA Formal Methods Symposium. Springer, 121–138.

[17] G. Frehse. 2005. PHAVer: Algorithmic verification of hybrid systems past HyTech. In HSCC. Springer, 258–273.

[18] G. Frehse, C. Le Guernic, A. Donzé, S. Cotton, R. Ray, O. Lebeltel, R. Ripado, A. Girard, T. Dang, and O. Maler. 2011. SpaceEx: Scalable verification of hybrid systems. In Proc. of CAV’11 (LNCS), Vol. 6806. Springer, 379–395.

[19] Eduardo Gallestey and Peter Hokayem. 2019. Lecture notes in Nonlinear Systems and Control.

[20] T. A. Henzinger, P. W. Kopke, A. Puri, and P. Varaiya. 1998. What’s decidable about hybrid automata? Journal of Computer and System Sciences 57, 1 (1998), 94–124.

[21] Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. 2015. Distilling the knowledge in a neural network. CoRR abs/1503.02531 (2015).

[22] C. Huang, X. Chen, W. Lin, Z. Yang, and X. Li. 2017. Probabilistic safety verification of stochastic hybrid systems using barrier certificates. TECS 16, 5s (2017), 186.

[23] X. Huang, M. Kwiatkowska, S. Wang, and M. Wu. 2017. Safety verification of deep neural networks. In International Conference on Computer Aided Verification. Springer, 3–29.

[24] Radoslav Ivanov, James Weimer, Rajeev Alur, George J. Pappas, and Insup Lee. 2018. Verisig: Verifying safety properties of hybrid systems with neural network controllers. arXiv preprint arXiv:1811.01828 (2018).

[25] G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer. 2017. Reluplex: An efficient SMT solver for verifying deep neural networks. In International Conference on Computer Aided Verification. Springer, 97–117.

[26] S. Kong, S. Gao, W. Chen, and E. M. Clarke. 2015. dReach: δ-reachability analysis for hybrid systems. In Proc. of TACAS’15 (LNCS), Vol. 9035. Springer, 200–205.

[27] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. 2016. Continuous control with deep reinforcement learning. CoRR abs/1509.02971 (2016).

[28] George G. Lorentz. 2013. Bernstein Polynomials. American Mathematical Soc.

[29] J. Lygeros, C. Tomlin, and S. Sastry. 1999. Controllers for reachability specifications for hybrid systems. Automatica 35, 3 (1999), 349–370.

[30] K. Makino and M. Berz. 2005. Verified global optimization with taylor model-based range bounders. Transactions on Computers 11, 4 (2005), 1611–1618.

[31] J. D. Meiss. 2007. Differential Dynamical Systems. SIAM publishers.

[32] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, et al. 2015. Human-level control through deep reinforcement learning. Nature 518, 7540 (2015), 529.

[33] Yunpeng Pan, Ching-An Cheng, Kamil Saigol, Keuntaek Lee, Xinyan Yan, Evangelos Theodorou, and Byron Boots. 2018. Agile autonomous driving using end-to-end deep imitation learning. Proceedings of Robotics: Science and Systems. Pittsburgh, Pennsylvania (2018).

[34] S. Prajna and A. Jadbabaie. 2004. Safety verification of hybrid systems using barrier certificates. In HSCC. Springer, 477–492.

[35] H. L. Royden. 1968. Real Analysis. Krishna Prakashan Media.

[36] W. Ruan, X. Huang, and M. Kwiatkowska. 2018. Reachability analysis of deep neural networks with provable guarantees. arXiv preprint arXiv:1805.02242 (2018).

[37] Georgi V. Smirnov. 2002. Introduction to the Theory of Differential Inclusions. Vol. 41. American Mathematical Soc.

[38] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus. 2013. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199 (2013).

[39] W. Xiang and T. T. Johnson. 2018. Reachability analysis and safety verification for neural network control systems. arXiv preprint arXiv:1805.09944 (2018).

[40] Z. Yang, C. Huang, X. Chen, W. Lin, and Z. Liu. 2016. A linear programming relaxation based approach for generating barrier certificates of hybrid systems. In FM. Springer, 721–738.

[41] Yuichi Yoshida and Takeru Miyato. 2017. Spectral norm regularization for improving the generalizability of deep learning. arXiv preprint arXiv:1705.10941 (2017).

[42] F. Zhao. 1992. Automatic Analysis and Synthesis of Controllers for Dynamical Systems Based on Phase-Space Knowledge. Ph.D. Dissertation. Massachusetts Institute of Technology. 