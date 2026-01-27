# How Many Bits Does it Take to Quantize Your  Neural Network?

<center>Mirco Giacobbe, Thomas A. Henzinger, and Mathias Lechner</center>

Quantization converts neural networks into low-bit fixedpoint computations which can be carried out by efficient integer-only hardware, and is standard practice for the deployment of neural networks on real-time embedded devices. However, like their real-numbered counterpart, quantized networks are not immune to malicious misclassification caused by adversarial attacks. We investigate how quantization affects a network’s robustness to adversarial attacks, which is a formal verification question. We show that neither robustness nor nonrobustness are monotonic with changing the number of bits for the representation and, also, neither are preserved by quantization from a realnumbered network. For this reason, we introduce a verification method for quantized neural networks which, using SMT solving over bit-vectors, accounts for their exact, bit-precise semantics. We built a tool and analyzed the effect of quantization on a classifier for the MNIST dataset. We demonstrate that, compared to our method, existing methods for the analysis of real-numbered networks often derive false conclusions about their quantizations, both when determining robustness and when detecting attacks, and that existing methods for quantized networks often miss attacks. Furthermore, we applied our method beyond robustness, showing how the number of bits in quantization enlarges the gender bias of a predictor for students’ grades.

## Introduction

Deep neural networks are powerful machine learning models, and are becoming increasingly popular in software development. Since recent years, they have pervaded our lives: think about the language recognition system of a voice assistant, the computer vision employed in face recognition or self driving, not to talk about many decision-making tasks that are hidden under the hood. However, this also subjects them to the resource limits that real-time embedded devices impose. Mainly, the requirements are low energy consumption, as they often run on batteries, and low latency, both to maintain user engagement and to effectively interact with the physical world. This translates into specializing our computation by reducing the memory footprint and instruction set, to minimize cache misses and avoid costly hardware operations. For this purpose, quantization compresses neural networks, which are traditionally run over 32-bit floating-point arithmetic, into computations that require bit-wise and integeronly arithmetic over small words, e.g., 8 bits. Quantization is the standard technique for the deployment of neural networks on mobile and embedded devices, and is implemented in TensorFlow Lite [13]. In this work, we investigate the robustness of quantized networks to adversarial attacks and, more generally, formal verification questions for quantized neural networks.

Adversarial attacks are a well-known vulnerability of neural networks [24]. For instance, a self-driving car can be tricked into confusing a stop sign with a speed limit sign [9], or a home automation system can be commanded to deactivate the security camera by a voice reciting poetry [22]. The attack is carried out by superposing the innocuous input with a crafted perturbation that is imperceptible to humans. Formally, the attack lies within the neighborhood of a known-to-be-innocuous input, according to some notion of distance. The fraction of samples (from a large set of test inputs) that do not admit attacks determines the robustness of the network. We ask ourselves how quantization affects a network’s robustness or, dually, how many bits it takes to ensure robustness above some specific threshold. This amounts to proving that, for a set of given quantizations and inputs, there does not exists an attack, which is a formal verification question.

The formal verification of neural networks has been addressed either by overapproximating—as happens in abstract interpretation—the space of outputs given a space of attacks, or by searching—as it happens in SMT-solving—for a variable assignment that witnesses an attack. The first category include methods that relax the neural networks into computations over interval arithmetic [20], treat them as hybrid automata [27], or abstract them directly by using zonotopes, polyhedra [10], or tailored abstract domains [23]. Overapproximationbased methods are typically fast, but incomplete: they prove robustness but do not produce attacks. On the other hand, methods based on local gradient descent have turned out to be effective in producing attacks in many cases [16], but sacrifice formal completeness. Indeed, the search for adversarial attack is NPcomplete even for the simplest (i.e., ReLU) networks [14], which motivates the rise of methods based on Satisfiability Modulo Theory (SMT) and Mixed Integer Linear Programming (MILP). SMT-solvers have been shown not to scale beyond toy examples (20 hidden neurons) on monolithic encodings [21], but today’s specialized techniques can handle real-life benchmarks such as, neural networks for the MNIST dataset. Specialized tools include DLV [12], which subdivides the problem into smaller SMT instances, and Planet [8], which combines different SAT and LP relaxations. Reluplex takes a step further augmenting LP-solving with a custom calculus for ReLU networks [14]. At the other end of the spectrum, a recent MILP formulation turned out effective using off-the-shelf solvers [25]. Moreover, it formed the basis for Sherlock [7], which couples local search and MILP, and for a specialized branch and bound algorithm [4].

All techniques mentioned above do not reason about the machine-precise semantics of the networks, neither over floating- nor over fixed-point arithmetic, but reason about a real-number relaxation. Unfortunately, adversarial attacks computed over the reals are not necessarily attacks on execution architectures, in particular, for quantized networks implementations. We show, for the first time, that attacks and, more generally, robustness and vulnerability to attacks do not always transfer between real and quantized networks, and also do not always transfer monotonically with the number of bits across quantized networks. Verifying the real-valued relaxation of a network may lead scenarios where

1. specifications are fulfilled by the real-valued network but not for its quantized implementation (false negative), 
2. specifications are violated by the real-valued network but fulfilled by its quantized representation (false negatives), or
3. counterexamples witnessing that the real-valued network violated the specification, but do not witness a violation for the quantized network (invalid counterexamples/attacks).

More generally, we show that all three phenomena can occur non-monotonically with the precision in the numerical representation. In other words, it may occur that a quantized network fulfills a specification while both a higher and a lower bits quantization violate it, or that the first violates it and both the higher and lower bits quantizations fulfill it; moreover, specific counterexamples may not transfer monotonically across quantizations.

The verification of real-numbered neural networks using the available methods is inadequate for the analysis of their quantized implementations, and the analysis of quantized neural networks needs techniques that account for their bit-precise semantics. Recently, a similar problem has been addressed for binarized neural networks, through SAT-solving [18]. Binarized networks represent the special case of 1-bit quantizations. For many-bit quantizations, a method based on gradient descent has been introduced recently [28]. While efficient (and sound), this method is incomplete and may produce false negatives.

The verification of real-numbered neural networks using the available methods is inadequate for the analysis of their quantized implementations, and the analysis of quantized neural networks needs techniques that account for their bit-precise semantics. Recently, a similar problem has been addressed for binarized neural networks, through SAT-solving [18]. Binarized networks represent the special case of 1-bit quantizations. For many-bit quantizations, a method based on gradient descent has been introduced recently [28]. While efficient (and sound), this method is incomplete and may produce false negatives.

We measured the robustness to attacks of a neural classifier involving 890 neurons and trained on the MNIST dataset (handwritten digits), for quantizations between 6 and 10 bits. First, we demonstrated that Boolector, off-the-shelf and using our balanced SMT encoding, can compute every attack within 16 hours, with a median time of 3h 41m, while timed-out on all instances beyond 6 bits using a standard linear encoding. Second, we experimentally confirmed that both Reluplex and gradient descent for quantized networks can produce false conclusions about quantized networks; in particular, spurious results occurred consistently more frequently as the number of bits in quantization decreases. Finally, we discovered that, to achieve an acceptable level of robustness, it takes a higher bit quantization than is assessed by standard accuracy measures.

Lastly, we applied our method beyond the property of robustness. We also evaluate the effect of quantization upon the gender bias emerging from quantized predictors for students’ performance in mathematics exams. More precisely, we computed the maximum predictable grade gap between any two students with identical features except for gender. The experiment showed that a substantial gap existed and was proportionally enlarged by quantization: the lower the number bits the larger the gap.

We summarize our contribution in five points. 

- First, we show that the robustness of quantized neural networks is non-monotonic in the number of bits and is non-transferable from the robustness of their real-numbered counterparts. 
- Second, we introduce the first complete method for the verification of quantized neural networks. 
- Third, we demonstrate that our encoding, in contrast to standard encodings, enabled the state-of-the-art SMT-solver Boolector to verify quantized networks with hundreds of neurons. 
- Fourth, we also show that existing methods determine both robustness and vulnerability of quantized networks less accurately than our bit-precise approach, in particular for low-bit quantizations. 
- Fifth, we illustrate how quantization affects the robustness of neural networks, not only with respect to adversarial attacks, but also with respect to other verification questions, specifically fairness in machine learning.

## Conclusion

- We introduced the first **complete** method for the verification of quantized neural networks which, by SMT solving over bit-vectors, accounts for their bit-precise semantics. 
- We demonstrated, both theoretically and experimentally, that **bit-precise reasoning** is necessary to accurately ensure the robustness to adversarial attacks of a quantized network. 
- We showed that robustness and non-robustness are **non-monotonic** in the number of bits for the numerical representation and that, consequently, the analysis of high-bits or real-numbered networks may derive false conclusions about their lower-bits quantizations. 
- Experimentally, we confirmed that real-valued solvers produce many spurious results, especially on low-bit quantizations, and that also gradient descent may miss attacks. 
- Additionally, we showed that quantization indeed affects not only robustness, but also other properties of neural networks, such as fairness. 
- We also demonstrated that, using our balanced encoding, off-the-shelf SMT-solving can analyze networks with **hundreds of neurons** which, despite hitting the limits of current solvers, establishes an encouraging **baseline** for future research.

## References

1. Students performance in exams. https://www.kaggle.com/spscientist/students-performance-in-exams

2. Barocas, S., Hardt, M., Narayanan, A.: **Fairness in machine learning**. In: Proceeding of NIPS (2017)

3. Barrett, C., Conway, C.L., Deters, M., Hadarean, L., Jovanović, D., King, T., Reynolds, A., Tinelli, C.: **Cvc4**. In: International Conference on Computer Aided Verification. pp. 171–177. Springer (2011)

4. Bunel, R.R., Turkaslan, I., Torr, P.H.S., Kohli, P., Mudigonda, P.K.: **A unified view of piecewise linear neural network verification**. In: NeurIPS. pp. 4795–4804 (2018)

5. De Moura, L., Bjørner, N.: **Z3: An efficient smt solver**. In: International Conference on Tools and Algorithms for the Construction and Analysis of Systems. pp. 337–340. Springer (2008)

6. Dutertre, B.: **Yices 2.2**. In: International Conference on Computer Aided Verification. pp. 737–744. Springer (2014)

7. Dutta, S., Jha, S., Sankaranarayanan, S., Tiwari, A.: **Output range analysis for deep feedforward neural networks**. In: NFM. Lecture Notes in Computer Science, vol. 10811, pp. 121–138. Springer (2018)

8. Ehlers, R.: **Formal verification of piecewise linear feedforward neural networks**. In: ATVA. Lecture Notes in Computer Science, vol. 10482, pp. 269–286. Springer (2017)

9. Evtimov, I., Eykholt, K., Fernandes, E., Kohno, T., Li, B., Prakash, A., Rahmati, A., Song, D.: **Robust physical-world attacks on deep learning models**. arXiv preprint arXiv:1707.08945 (2017)

10. Gehr, T., Mirman, M., Drachsler-Cohen, D., Tsankov, P., Chaudhuri, S., Vechev, M.T.: **AI2: Safety and robustness certification of neural networks with abstract interpretation**. In: IEEE Symposium on Security and Privacy. pp. 3–18. IEEE (2018)

11. Hadarean, L., Hyvarinen, A., Niemetz, A., Reger, G.: Smt-comp 2019. https://smt-comp.github.io/2019/results (2019)

12. Huang, X., Kwiatkowska, M., Wang, S., Wu, M.: **Safety verification of deep neural networks**. In: CAV (1). Lecture Notes in Computer Science, vol. 10426, pp. 3–29. Springer (2017)

13. Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A.G., Adam, H., Kalenichenko, D.: **Quantization and training of neural networks for efficient integer-arithmetic-only inference**. In: CVPR. pp. 2704–2713. IEEE Computer Society (2018)

14. Katz, G., Barrett, C.W., Dill, D.L., Julian, K., Kochenderfer, M.J.: **Reluplex: An efficient SMT solver for verifying deep neural networks**. In: CAV (1). Lecture Notes in Computer Science, vol. 10426, pp. 97–117. Springer (2017)

15. Krizhevsky, A., Hinton, G.: **Convolutional deep belief networks on cifar-10**. Unpublished manuscript 40(7) (2010)

16. Moosavi-Dezfooli, S., Fawzi, A., Frossard, P.: **Deepfool: A simple and accurate method to fool deep neural networks**. In: CVPR. pp. 2574–2582. IEEE Computer Society (2016)

17. Nair, V., Hinton, G.E.: **Rectified linear units improve restricted boltzmann machines**. In: ICML. pp. 807–814. Omnipress (2010)

18. Narodytska, N., Kasiviswanathan, S.P., Ryzhyk, L., Sagiv, M., Walsh, T.: **Verifying properties of binarized deep neural networks**. In: AAAI. pp. 6615–6624. AAAI Press (2018)

19. Niemetz, A., Preiner, M., Biere, A.: Boolector 2.0. JSAT 9, 53–58 (2014)

20. Pulina, L., Tacchella, A.: **An abstraction-refinement approach to verification of artificial neural networks**. In: CAV. Lecture Notes in Computer Science, vol. 6174, pp. 243–257. Springer (2010)

21. Pulina, L., Tacchella, A.: **Challenging SMT solvers to verify neural networks**. AI Commun. 25(2), 117–135 (2012)

22. Schönherr, L., Kohls, K., Zeiler, S., Holz, T., Kolossa, D.: **Adversarial attacks against automatic speech recognition systems via psychoacoustic hiding**. In: accepted for Publication, NDSS (2019)

23. Singh, G., Gehr, T., Püschel, M., Vechev, M.T.: **An abstract domain for certifying neural networks**. In: POPL. ACM (2019)

24. Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I.J., Fergus, R.: **Intriguing properties of neural networks**. CoRR abs/1312.6199 (2013)

25. Tjeng, V., Xiao, K.Y., Tedrake, R.: **Evaluating robustness of neural networks with mixed integer programming** (2018)

26. Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., Madry, A.: **Robustness may be at odds with accuracy**. In: International Conference on Learning Representations (2019)

27. Xiang, W., Tran, H., Johnson, T.T.: **Output reachable set estimation and verification for multilayer neural networks**. IEEE Trans. Neural Netw. Learning Syst. 29(11), 5777–5783 (2018)

28. Zhao, Y., Shumailov, I., Mullins, R., Anderson, R.: **To compress or not to compress: Understanding the interactions between adversarial attacks and neural network compression**. In: SysML Conference (2019)
