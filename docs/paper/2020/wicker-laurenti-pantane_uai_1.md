# Probabilistic Safety for Bayesian Neural Networks

<center>Matthew Wicker∗, Luca Laurenti∗,  Andrea Patane∗, Marta Kwiatkowska</center>

## Abstract

- We study probabilistic safety for Bayesian Neural Networks (BNNs) under adversarial input perturbations. 
- Given a compact set of input points, $T \subseteq \mathbb{R}^m$, we study the probability w.r.t. the BNN posterior that all the points in $T$ are mapped to the same region $S$ in the output space. 
- In particular, this can be used to evaluate the probability that a network sampled from the BNN is vulnerable to adversarial attacks. 
- We rely on relaxation techniques from **non-convex optimization** to develop a method for computing a lower bound on probabilistic safety for BNNs, deriving explicit procedures for the case of interval and linear function propagation techniques. 
- We apply our methods to BNNs trained on a regression task, airborne collision avoidance, and MNIST, empirically showing that our approach allows one to certify probabilistic safety of BNNs with millions of parameters.

## INTRODUCTION

Although Neural Networks (NNs) have recently achieved state-of-the-art performance [14], they are susceptible to several vulnerabilities [3], with adversarial examples being one of the most prominent among them [28]. Since adversarial examples are arguably intuitively related to uncertainty [17], Bayesian Neural Networks (BNNs), i.e. NNs with a probability distribution placed over their weights and biases [24], have recently been proposed as a potentially more robust learning paradigm [6]. While retaining the advantages intrinsic to deep learning (e.g. representation learning), BNNs also enable principled evaluation of model uncertainty, which can be taken into account at prediction time to enable safe decision making.

Many techniques have been proposed for the evaluation of the robustness of BNNs, including generalization of NN gradient-based adversarial attacks to BNN posterior distribution [19], statistical verification techniques for adversarial settings [7], as well as pointwise (i.e. for a specific test point $x^*$) uncertainty evaluation techniques [27]. However, to the best of our knowledge, methods that formally (i.e., with certified bounds) give guarantees on the behaviour of BNNs against adversarial input perturbations in probabilistic settings are missing.

In this paper we aim at analysing probabilistic safety for BNNs. Given a BNN, a set $T \subseteq \mathbb{R}^m$ in the input space, and a set $S \subseteq \mathbb{R}^n$ in the output space, probabilistic safety is defined as the probability that for all $x \in T$ the prediction of the BNN is in $S$. In adversarial settings, the input region $T$ is built around the neighborhood of a given test point $x^*$, while the output region $S$ is defined around the BNN prediction on $x^*$, so that probabilistic safety translates into computing the probability that adversarial perturbations of $x^*$ cause small variations in the BNN output. Note that probabilistic safety represents a probabilistic variant of the notion of safety commonly used to certify deterministic NNs [26].