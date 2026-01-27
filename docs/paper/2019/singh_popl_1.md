# An Abstract Domain for Certifying Neural Networks

GAGANDEEP SINGH, ETH Zurich, Switzerland<br/>TIMON GEHR, ETH Zurich, Switzerland<br/>MARKUS PÜSCHEL, ETH Zurich, Switzerland<br/>MARTIN VECHEV, ETH Zurich, Switzerland

We present a novel method for scalable and precise certification of deep neural networks. The key technical insight behind our approach is a new abstract domain which combines floating point polyhedra with intervals and is equipped with abstract transformers specifically tailored to the setting of neural networks. Concretely, we introduce new transformers for affine transforms, the rectified linear unit (ReLU), sigmoid, tanh, and maxpool functions.

We implemented our method in a system called DeepPoly and evaluated it extensively on a range of datasets, neural architectures (including defended networks), and specifications. Our experimental results indicate that DeepPoly is more precise than prior work while scaling to large networks.

We also show how to combine DeepPoly with a form of **abstraction refinement** based on **trace partitioning**. This enables us to prove, for the first time, the robustness of the network when the input image is subjected to complex perturbations such as rotations that employ linear interpolation.

## INTRODUCTION

Over the last few years, deep neural networks have become increasingly popular and have now started penetrating **safety critical domains** such as autonomous driving [Bojarski et al. 2016] and medical diagnosis [Amato et al. 2013] where they are often relied upon for making important decisions. As a result of this widespread adoption, it has become even more important to ensure that neural networks behave reliably and as expected. Unfortunately, reasoning about these systems is challenging due to their "black box" nature: it is difficult to understand what the network does since it is typically parameterized with thousands or millions of real-valued weights that are hard to interpret. Further, it has been discovered that neural nets can sometimes be surprisingly brittle and exhibit non-robust behaviors, for instance, by classifying two very similar inputs (e.g., images that differ only in brightness or in one pixel) to different labels [Goodfellow et al. 2015].

To address the challenge of reasoning about neural networks, recent research has started exploring new methods and systems which can automatically prove that a given network satisfies a specific property of interest (e.g., robustness to certain perturbations, pre/post conditions). State-of-the-art works include methods based on **SMT solving** [Katz et al. 2017], **linear approximations** [Weng et al. 2018], and **abstract interpretation** [Gehr et al. 2018; Mirman et al. 2018; Singh et al. 2018a].

Despite the progress made by these works, more research is needed to reach the point where we are able to solve the overall neural network reasoning challenge successfully. In particular, we still lack an analyzer that can scale to large networks, is able to handle popular neural architectures (e.g., feedforward, convolutional), and yet is sufficiently precise to prove relevant properties required by applications. For example, the work by Katz et al. [2017] is precise yet can only handle very small networks. At the same time, Gehr et al. [2018] can analyze larger networks than Katz et al. [2017], but relies on existing **generic abstract domains** which either do not scale to larger neural networks (such as **Convex Polyhedra** [Cousot and Halbwachs 1978]) or are too imprecise (e.g., **Zonotope** [Ghorbal et al. 2009]). Recent work by Weng et al. [2018] scales better than Gehr et al. [2018] but only handles feedforward networks and cannot handle the widely used convolutional networks. Both Katz et al. [2017] and Weng et al. [2018] are in fact **unsound for floating point arithmetic**, which is heavily used in neural nets, and thus they can suffer from false negatives. Recent work by Singh et al. [2018a] handles feedforward and convolutional networks and is sound for floating point arithmetic, however, as we demonstrate experimentally, it can lose significant precision when dealing with larger perturbations.

In this work, we propose a new method and system, called DeepPoly, that makes a step forward in addressing the challenge of verifying neural networks with respect to both scalability and precision. The key technical idea behind DeepPoly is a novel abstract interpreter specifically tailored to the setting of neural networks. Concretely, our abstract domain is a combination of floating-point polyhedra with intervals, coupled with abstract transformers for common neural network functions such as affine transforms, the rectified linear unit (ReLU), sigmoid and tanh activations, and the maxpool operator. These abstract transformers are carefully designed to exploit key properties of these functions and balance analysis scalability and precision. As a result, DeepPoly is more precise than Weng et al. [2018], Gehr et al. [2018] and Singh et al. [2018a], yet can handle large convolutional networks and is sound for floating point arithmetic.

?> 注意到在这个语境下，Sound是指不会把不robust的判为robust（即没有假阴性）。

## RELATED WORK

We already extensively discussed the works that are most closely related throughout the paper, here we additionally elaborate on several others.

**Generating adversarial examples.** There is considerable interest in constructing examples that make the neural network misclassify an input. 

- Nguyen et al. [2015] find adversarial examples without starting from a test point, 
- Tabacof and Valle [2016] use random perturbations for generating adversarial examples, 
- Sabour et al. [2015] demonstrate non-robustness of intermediate layers, and 
- Grosse et al. [2016] generate adversarial examples for malware classification. 
- Pei et al. [2017a] systematically generate adversarial examples covering all neurons in the network. 
- Bastani et al. [2016] **under-approximate** the behavior of the network under $L_∞$-norm based perturbation and formally define metrics of adversarial frequency and adversarial severity to evaluate the robustness of a neural network against adversarial attack.

**Formal verification of neural network robustness.** Existing formal verifiers of neural network robustness can be broadly classified as either complete or incomplete. Complete verifiers do not have false positives but have limited scalability and cannot handle neural networks containing more than a few hundred hidden units whereas incomplete verifiers approximate for better scalability. 

- Complete verifiers are based on **SMT** solving [Ehlers 2017; Katz et al. 2017], **mixed integer linear programming** [Tjeng and Tedrake 2017] or **input** refinement [Wang et al. 2018] 
- whereas existing incomplete verifiers are based on **duality** [Dvijotham et al. 2018; Raghunathan et al. 2018], **abstract interpretation** [Gehr et al. 2018; Singh et al. 2018a], and **linear approximations** [Weng et al. 2018; Wong and Kolter 2018]. 

We note that although our verifier is designed to be incomplete for better scalability, it can be made complete by refining the input iteratively.

?> 怎么refine？

**Adversarial training.** There is growing interest in adversarial training where neural networks are trained against a model of adversarial attacks. 

- Gu and Rigazio [2014] add Gaussian noise to the training set and remove it statistically for defending against adversarial examples. 
- The approach of Goodfellow et al. [2015] first generates adversarial examples misclassified by neural networks and then designs a defense against this attack by explicitly training against perturbations generated by the attack. 
- Madry et al. [2018] shows that training against an optimal attack also guards against nonoptimal attacks. 
- While this was effective in experiments, Carlini et al. [2017] demonstrated an attack for the safety-critical problem of ground-truthing, where this defense occasionally exacerbated the problem. 
- Mirman et al. [2018] train neural networks against adversarial attacks using abstract transformers for the Zonotope domain. 

As mentioned earlier, our abstract transformers can be plugged into such a framework to potentially improve the training results.

## CONCLUSION

We introduced a new method for certifying deep neural networks which balances analysis precision and scalability. The core idea is an **abstract domain** based on floating point **polyhedra and intervals** equipped with **abstract transformers** specifically designed for common neural network functions such as affine transforms, ReLU, sigmoid, tanh, and maxpool. These abstract transformers enable us to soundly handle both, feed-forward and convolutional networks.

We implemented our method in an analyzer, called DeepPoly, and evaluated it extensively on a wide range of networks of different sizes including defended and undefended networks. Our experimental results demonstrate that DeepPoly is more precise than prior work yet can handle large networks.

We also showed how to use DeepPoly to prove, for the first time, the robustness of a neural network when the input image is perturbed by **complex transformations** such as rotations employing linear interpolation.

We believe this work is a promising step towards more effective reasoning about deep neural networks and a useful building block for proving interesting specifications as well as other applications of analysis (for example, training more robust networks).