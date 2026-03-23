# A Mathematical Introduction to Logic【Herbert B. Enderton】

## Preface

This book, like the first edition, presents the basic concepts and results of logic: the topics are **proofs**, **truth**, and **computability**. As before, the presentation is directed toward the reader with some mathematical background and interests. In this revised edition, in addition to numerous "local" changes, there are three "global" ways in which the presentation has been changed:

First, I have attempted to make the material more *accessible* to the typical undergraduate student. In the main development, I have tried not to take for granted information or insights that might be unavailable to a junior-level mathematics student.

Second, for the instructor who wants to fit the book to his or her course, the organization has been made more *flexible*. Footnotes at the beginning of many of the sections indicate optional paths the instructor — or the independent reader — might choose to take.

Third, theoretical *computer science* has influenced logic in recent years, and some of that influence is reflected in this edition. Issues of computability are taken more seriously. Some material on finite models has been incorporated into the text.

The book is intended to serve as a textbook for an introductory mathematics course in logic at the junior-senior level. The objectives are to present the important concepts and theorems of logic and to explain their significance and their relationship to the reader's other mathematical work.

As a text, the book can be used in courses anywhere from a quarter to a year in length. In one quarter, I generally reach the material on models of first-order theories (Section 2.6). The extra time afforded by a semester would permit some glimpse of undecidability, as in Section 3.0. In a second term, the material of Chapter 3 (on undecidability) can be more adequately covered.

The book is intended for the reader who has not studied logic previously, but who has some experience in mathematical reasoning. There are no specific prerequisites aside from a willingness to function at a certain level of abstraction and rigor. There is the inevitable use of basic set theory. Chapter 0 gives a concise summary of the set theory used. One should not begin the book by studying this chapter; it is instead intended for reference if and when the need arises. The instructor can adjust the amount of set theory employed; for example it is possible to avoid cardinal numbers completely (at the cost of losing some theorems). The book contains some examples drawn from abstract algebra. But they are just examples, and are not essential to the exposition. The later chapters (Chapter 3 and 4) tend to be more demanding of the reader than are the earlier chapters.

Induction and recursion are given a more extensive discussion (in Section 1.4) than has been customary. I prefer to give an informal account of these subjects in lectures and have a precise version in the book rather than to have the situation reversed.

Exercises are given at the end of nearly all the sections. If the exercise bears a boldface numeral, then the results of that exercise are used in the exposition in the text. Unusually challenging exercises are marked with an asterisk.

I cheerfully acknowledge my debt to my teachers, a category in which I include also those who have been my colleagues or students. I would be pleased to receive comments and corrections from the users of this book. The Web site for the book can be found at http://www.math.ucla.edu/~hbe/amil.

## Introduction

**Symbolic logic** is a mathematical model of deductive thought. Or at least that was true originally; as with other branches of mathematics it has grown beyond the circumstances of its birth. Symbolic logic is a model in much the same way that modern probability theory is a model for situations involving chance and uncertainty.

How are models constructed? You begin with a real-life object, for example an airplane. Then you select some features of this original object to be represented in the model, for example its shape, and others to be ignored, for example its size. And then you build an object that is like the original in some ways (which you call essential) and unlike it in others (which you call irrelevant). Whether or not the resulting model meets its intended purpose will depend largely on the selection of the properties of the original object to be represented in the model.

Logic is more abstract than airplanes. The real-life objects are certain "logically correct" deductions. For example,

> All men are mortal.
> Socrates is a man.
> Therefore, Socrates is mortal.

The validity of inferring the third sentence (the conclusion) from the first two (the assumptions) does not depend on special ==idiosyncrasies== of Socrates. The inference is justified by the form of the sentences rather than by empirical facts about mortality. It is not really important here what "mortal" means; it does matter what "all" means.

> Borogoves are mimsy whenever it is brillig.
> It is now brillig, and this thing is a borogove.
> Hence this thing is mimsy.

Again we can recognize that the third sentence follows from the first two, even without the slightest idea of what a mimsy borogove might look like.

Logically correct deductions are of more interest than the above ==frivolous== examples might suggest. In fact, axiomatic mathematics consists of many such deductions laid end to end. These deductions made by the working mathematician constitute real-life originals whose features are to be mirrored in our model.

**The logical correctness of these deductions is due to their form but is independent of their content**. This criterion is vague, but it is just this sort of vagueness that prompts us to turn to mathematical models. A major goal will be to give, within a model, a precise version of this criterion. The questions (about our model) we will initially be most concerned with are these:

1. What does it mean for one sentence to "follow logically" from certain others?
2. If a sentence does follow logically from certain others, what methods of *proof* might be necessary to establish this fact?
3. **Is there a gap between what we can *prove* in an axiomatic system (say for the natural numbers) and what is *true* about the natural numbers?**
4. What is the connection between logic and computability?

Actually we will present two models. The first (**sentential logic**) will be very simple and will be ==woefully== inadequate for interesting deductions. Its inadequacy stems from the fact that it preserves only some ==crude== properties of real-life deductions. The second model (**first-order logic**) is admirably suited to deductions encountered in mathematics. **When a working mathematician asserts that a particular sentence follows from the axioms of set theory, he or she means that this deduction can be translated to one in our model.**

This emphasis on mathematics has guided the choice of topics to include. This book does not venture into **many-valued logic**, **modal logic**, or **intuitionistic logic**, which represent different selections of properties of real-life deductions.

Thus far we have avoided giving away much information about what our model, first-order logic, is like. As brief hints, we now give some examples of the expressiveness of its formal language. First, take the English sentence that asserts the set-theoretic principle of extensionality, "If the same things are members of a first object as are members of a second object, then those objects are the same." This can be translated into our **first-order language** as
$$
\forall x \forall y (\forall z (z \in x \leftrightarrow z \in y) \rightarrow x = y).
$$
As a second example, we can translate the sentence familiar to calculus students, "For every positive number $\varepsilon$ there is a positive number $\delta$ such that for any $x$ whose distance from $a$ is less than $\delta$, the distance between $f(x)$ and $b$ is less than $\varepsilon$" as
$$
\forall \varepsilon (\varepsilon > 0 \rightarrow \exists \delta (\delta > 0 \land \forall x (\lang x,a\rang < \delta \rightarrow \lang f(x),b \rang < \varepsilon))).
$$
We have given some hints as to what we intend to do in this book. We should also correct some possible misimpressions by saying what we are not going to do. This book does not propose to teach the reader how to think. The word "logic" is sometimes used to refer to ==remedial== thinking, but not by us. The reader already knows how to think. Here are some intriguing concepts to think about.
