# Efficient Formal Safety Analysis of Neural Networks

<center>Shiqi Wang, Kexin Pei, Justin Whitehouse, Junfeng Yang, Suman Jana</center>

Neural networks are increasingly deployed in real-world safety-critical domains such as autonomous driving, aircraft collision avoidance, and malware detection. However, these networks have been shown to often mispredict on inputs with minor adversarial or even accidental perturbations. Consequences of such errors can be disastrous and even potentially fatal as shown by the recent Tesla autopilot crashes. Thus, there is an urgent need for formal analysis systems that can rigorously check neural networks for violations of different safety properties such as robustness against adversarial perturbations within a certain $L$-norm of a given image. An effective safety analysis system for a neural network must be able to either ensure that a safety property is satisfied by the network or find a counterexample, i.e., an input for which the network will violate the property. Unfortunately, most existing techniques for performing such analysis **struggle to scale** beyond very small networks and the ones that can scale to larger networks suffer from **high false positives** and cannot produce concrete counterexamples in case of a property violation. In this paper, we present a new efficient approach for rigorously checking different safety properties of neural networks that significantly outperforms existing approaches by multiple orders of magnitude. Our approach can check different safety properties and find concrete counterexamples for networks that are 10× larger than the ones supported by existing analysis techniques. We believe that our approach to estimating tight output bounds of a network for a given input range can also help improve the explainability of neural networks and guide the training process of more robust neural networks.

## Introduction

## Background

We build upon two prior works [10, 39] on using interval analysis and linear relaxations for analyzing neural networks. We briefly describe them and refer interested readers to [10, 39] for more details.

### Symbolic Interval Analysis

Interval arithmetic [33] is a flexible and efficient way of rigorously estimating the output ranges of a function given an input range by computing and propagating the output intervals for each operation in the function. However, naive interval analysis suffers from large overestimation errors as it ignores the input dependencies during interval propagation. 

To minimize such errors, Wang et al. [39] used **symbolic intervals** to keep track of dependencies by maintaining linear equations for upper and lower bounds for each ReLU and concretizing only for those ReLUs that demonstrate non-linear behavior for the given input intervals. Specifically, consider an intermediate ReLU node $z=\operatorname{Relu}(Eq)$, where $Eq$ denotes the symbolic representation (i.e., a closed-form equation) of the ReLU's input in terms of network inputs $X$ and $(l,u)$ denote the concrete lower and upper bounds of $Eq$, respectively. There are three possible output intervals that the ReLU node can produce depending on the bounds of $Eq$: 

1. $z=[Eq,Eq]$ when $l\geq 0$, 
2. $z=[0,0]$ when $u\leq 0$, or 
3. $z=[l,u]$ when $l<0<u$. 

Wang et al. will concretize the output intervals for this node only if the third case is feasible as the output in this case cannot be represented using a single linear equation.

### Bisection of Input Features

To further minimize overestimation, [39] also proposed an iterative refinement strategy involving repeated input bisection and output reunion. Consider a network $F$ taking $d$-dimensional input, and the $i$-th input feature interval is $X_{i}$ and network output interval is $F(X)$ where $X=\{X_{1},...,X_{d}\}$. A single bisection on $X_{i}$ will create two children: $X^{\prime}=\{X_{1},...,[\underline{X_{i}},\frac{\underline{X_{i}}+\overline{X_{i}}}{2}],...,X_{d}\}$ and $X^{\prime\prime}=\{X_{1},...,[\frac{\underline{X_{i}}+\overline{X_{i}}}{2},\overline{X_{i}}],...,X_{d}\}$. The reunion of the corresponding output intervals $F(X') \bigcup F(X'')$, will be tighter than the original output interval, i.e., $F(X')\bigcup F(X'')\subseteq F(X)$, as the Lipschitz continuity of the network ensures that the overestimation error decreases as the width of input interval becomes smaller. However, the efficiency of input bisection decreases drastically as the number of input dimensions increases.

### Linear Relaxation

Ehlers et al. [10] used linear relaxation of ReLU nodes to strictly over-approximate the non-linear constraints introduced by each ReLU. The generated linear constraints can then be efficiently solved using a linear solver to get bounds on the output of a neural network for a given input range. Consider the simple ReLU node taking input $z^{\prime}$ with an upper and lower bound $u$ and $l$ respectively and producing output $z$ as shown in Figure 1. Linear relaxation of such a node will use the following three linear constraints: (1) $z\geq 0$, (2) $z\geq z^{\prime}$, and (3) $z\leq \frac{u(z^{\prime}-l)}{u-l}$ to expand the feasible region to the green triangle from the two original piecewise linear components. The effectiveness of this approach heavily depends on how accurately $u$ and $l$ can be estimated. Unfortunately, Ehlers et al. [10] used naive interval propagation to estimate $u$ and $l$ leading to large overestimation errors. Furthermore, their approach cannot efficiently refine the estimated bounds and thus cannot benefit from increasing computing power.

## References

[1] NAVAIR plans to install ACAS Xu on MQ-4C fleet. https://www.flightglobal.com/news/articles/navair-plans-to-install-acas-xu-on-mq-4c-fleet-444989/.

[2] Nvidia-Autopilot-Keras. https://github.com/Observer07/Nvidia-Autopilot-Keras.

[3] Tesla's autopilot was involved in another deadly car crash. https://www.wired.com/story/tesla-autopilot-self-driving-crash-california/.

[4] Using Deep Learning to Predict Steering Angles. https://github.com/udacity/self-driving-car.

[5] D.Arp, M. Spreitzenbarth, M. Hubner, H. Gascon, K. Rieck, and C. Siemens. Drebin: Effective and explainable detection of android malware in your pocket. In Proceedings of the Network and Distributed System Security Symposium, volume 14, pages 23-26, 2014.

[6] M.Bojarski, D.Del Testa, D.Dworakowski, B.Firner, B.Flepp, P.Goyal, L.D.Jackel, M.Monfort, U. Muller, J. Zhang, et al. End to end learning for self-driving cars. IEEE Intelligent Vehicles Symposium, 2017.

[7] S. Dutta, S. Jha, S. Sankaranarayanan, and A. Tiwari. Output range analysis for deep feedforward neural networks. In NASA Formal Methods Symposium, pages 121-138. Springer, 2018.

[8] K. Dvijotham, S. Gowal, R. Stanforth, R. Arandjelovic, B.O'Donoghue, J. Uesato, and P. Kohli. Training verified learners with learned verifiers. arXiv preprint arXiv:1805.10265, 2018.

[9] K. Dvijotham, R. Stanforth, S. Gowal, T. Mann, and P. Kohli. A dual approach to scalable verification of deep networks. The Conference on Uncertainty in Artificial Intelligence, 2018.

[10] R. Ehlers. Formal verification of piece-wise linear feed-forward neural networks. 15th International Symposium on Automated Technology for Verification and Analysis, 2017.

[11] R. Eldan. A polynomial number of random points does not determine the volume of a convex body. Discrete & Computational Geometry, 46(1):29-47, 2011.

[12] M. Fischetti and J. Jo. Deep neural networks as 0-1 mixed integer linear programs: A feasibility study. arXiv preprint arXiv:1712.06174, 2017.

[13] T. Gehr, M. Mirman, D. Drachsler-Cohen, P. Tsankov, S. Chaudhuri, and M. Vechev. Ai 2: Safety and robustness certification of neural networks with abstract interpretation. In IEEE Symposium on Security and Privacy, 2018.

[14] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. International Conference on Learning Representations, 2015.

[15] X. Huang, M. Kwiatkowska, S. Wang, and M. Wu. Safety verification of deep neural networks. In International Conference on Computer Aided Verification, pages 3-29. Springer, 2017.

[16] K.D.Julian, J. Lopez, J.S.Brush, M.P.Owen, and M.J. Kochenderfer. Policy compression for aircraft collision avoidance systems. In 35th Digital Avionics Systems Conference, pages 1-10. IEEE, 2016.

[17] G.Katz, C.Barrett, D.Dill, K.Julian, and M. Kochenderfer. Reluplex: An efficient smt solver for verifying deep neural networks. International Conference on Computer Aided Verification, 2017.

[18] G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer. Towards proving the adversarial robustness of deep neural networks. 1st Workshop on Formal Verification of Autonomous Vehicles, 2017.

[19] M. J. Kochenderfer, J. E. Holland, and J. P. Chryssanthacopoulos. Next-generation airborne collision avoidance system. Technical report, Massachusetts Institute of Technology-Lincoln Laboratory Lexington United States, 2012.

[20] P. W. Koh and P. Liang. Understanding black-box predictions via influence functions. International Conference on Machine Learning, 2017.

[21] Y. LeCun. The mnist database of handwritten digits. http://yann. lecun. com/exdb/mnist/, 1998.

[22] M. Lecuyer, V. Atlidakis, R. Geambasu, H. Daniel, and S. Jana. Certified robustness to adversarial examples with differential privacy. arXiv preprint arXiv:1802.03471, 2018.

[23] J.Li, W. Monroe, and D. Jurafsky. Understanding neural networks through representation erasure. arXiv preprint arXiv:1612.08220, 2016.

[24] M. Mirman, T. Gehr, and M. Vechev. Differentiable abstract interpretation for provably robust neural networks. In International Conference on Machine Learning, pages 3575-3583, 2018.

[25] G.F. Montufar, R. Pascanu, K. Cho, and Y. Bengio. On the number of linear regions of deep neural networks. In Advances in neural information processing systems, pages 2924-2932, 2014.

[26] M.T. Notes. Airborne collision avoidance system x. MIT Lincoln Laboratory, 2015.

[27] R.Pascanu, G. Montufar, and Y. Bengio. On the number of response regions of deep feed forward networks with piece-wise linear activations. Advances in neural information processing systems, 2013.

[28] J. Peck, J. Roels, B. Goossens, and Y. Saeys. Lower bounds on the robustness to adversarial perturbations. In Advances in Neural Information Processing Systems, pages 804-813, 2017.

[29] K. Pei, Y. Cao, J. Yang, and S. Jana. Deepxplore: Automated whitebox testing of deep learning systems. In 26th Symposium on Operating Systems Principles, pages 1-18. ACM, 2017.

[30] K. Pei, Y. Cao, J. Yang, and S. Jana. Towards practical verification of machine learning: The case of computer vision systems. arXiv preprint arXiv:1712.01785, 2017.

[31] L. Pulina and A. Tacchella. An abstraction-refinement approach to verification of artificial neural networks. In International Conference on Computer Aided Verification, pages 243-257. Springer, 2010.

[32] A. Raghunathan, J. Steinhardt, and P. Liang. Certified defenses against adversarial examples. International Conference on Learning Representations, 2018.

[33] M.J.C. Ramon E. Moore, R. Baker Kearfott. Introduction to Interval Analysis. SIAM, 2009.

[34] A. Shrikumar, P. Greenside, and A. Kundaje. Learning important features through propagating activation differences. International Conference on Machine Learning, 2017.

[35] M. Spreitzenbarth, F. Freiling, F. Echtler, T. Schreck, and J. Hoffmann. Mobile-sandbox: having a deeper look into android applications. In 28th Annual ACM Symposium on Applied Computing, pages 1808-1815. ACM, 2013.

[36] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. International Conference on Learning Representations, 2013.

[37] Y. Tian, K. Pei, S. Jana, and B. Ray. DeepTest: Automated testing of deep-neural-network-driven autonomous cars. In 40th International Conference on Software Engineering, 2018.

[38] V.Tjeng, K. Xiao, and R. Tedrake. Evaluating robustness of neural networks with mixed integer programming. arXiv preprint arXiv:1711.07356, 2017.

[39] S. Wang, K. Pei, W. Justin, J. Yang, and S. Jana. **Formal security analysis of neural networks using symbolic intervals**. 27th USENIX Security Symposium, 2018.

[40] T.-W. Weng, H. Zhang, H. Chen, Z. Song, C.-J. Hsieh, D. Boning, I. S. Dhillon, and L. Daniel. Towards fast computation of certified robustness for relu networks. arXiv preprint arXiv:1804.09699, 2018.

[41] T.-W. Weng, H. Zhang, P.-Y. Chen, J. Yi, D. Su, Y. Gao, C.-J. Hsieh, and L. Daniel. Evaluating the robustness of neural networks: An extreme value theory approach. International Conference on Learning Representations, 2018.

[42] E. Wong and J. Z. Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. International Conference on Machine Learning, 2018.

[43] E. Wong, F. Schmidt, J. H. Metzen, and J. Z. Kolter. Scaling provable adversarial defenses. Advances in Neural Information Processing Systems, 2018.

[44] H. Zhang, T.-W. Weng, P.-Y. Chen, C.-J. Hsieh, and L. Daniel. Efficient neural network robustness certification with general activation functions. Advances in Neural Information Processing Systems, 2018.