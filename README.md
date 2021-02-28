# Radio-Link-Frequency-Assignment-Problem
📻📡 Solving the Constraint Satisfaction Radio Link Frequency Assignment Problem, using FC and MAC algorithms with the dom/wdeg heuristic. An attempt to solve it with MIN-CONFLICTS algorithm and some comparisons are also made. 

### Introduction to the Problem

Information by: https://miat.inrae.fr/schiex/rlfap.shtml

The Radio Link frequency Assignment Problem consists in assigning frequencies to a set of radio links defined between pairs of sites in order to avoid interferences. Each radio link is represented by a variable whose domain is the set of all frequences that are available for this link. The essential constraints involve two variables F1 and F2:
|F1-F2|> k12

The two variables represent two radio links which are "close" one to the other. The constant k12 depends on the position of the two links and also on the physical environment. It is obtained using a mathematical model of electromagnetic waves propagation which is still very "rough". Therefore, the constants k12 are not necessarily correct (it is possible that the minimum difference in frequency between F1 and F2 that does not yield interferences is actually different from k12). In practice, k12 is often overestimated in order to effectively guarantee the absence of interference. For each pair of sites, two frequencies must be assigned: one for the communications from A to B, the other one for the communications from B to A. The possibility of expressing constraints such as >|F1-F2|> k12 suffices to express the graph coloring problem and it is therefore clear that the RLFAP is NP-hard.

### Implementation

In RLFA.py, a class Parsing is used to parse the data from the files and initialize the CSP problem. A function check_constraints() is used as a checking function for the csp module.<br>
Each instance to be solved, consists of three files. A file with the prefix var contains the number of total variables of the csp problem in the first line, followed by the variables along with their domain ids. A file with the prefix dom contains the number of total domains of the csp problem in the first line, followed by the domain ids along with the actual domains. A file with the prefix ctr contains the number of total constraints of the csp problem in the first line, followed by the constraints in the format x y > k or x y = k, which means |x-y| > k ή |x-y| = k correspondingly.

The modules csp_dom_wdeg.py, search.py and utils.py contain code from [csp.py provided by AIMA](https://github.com/aimacode/aima-python).
The module csp_dom_wdeg.py contains the implementaion of FC, MAC and MIN-CONFLICTS algorithms and is modified in order to implement the dom/wdeg heuristic function described in F. Boussemart, F. Hemery, C. Lecoutre and L. Sais. Boosting Systematic Search by Weighting Constraints. Proc. of ECAI 2004, pages 146–150, 2004, available [here](http://www.frontiersinai.com/ecai/ecai2004/ecai04/pdf/p0146.pdf).

A main.py is provided in order to run the FC, MAC-AC3 and MIN-CONFLICTS algorithms for the 12 instances of the csp problem that are included.

### How to Run

Run with any python3 version, from python3.7 and above, for example:

```
>>> python3.7 main.py
```

### Inferences

An experimental comparison was made using the 12 inferences provided above. During this comparison, FC and MAC-AC3 algorithms used the dom/wdeg heuristic function to choose the next variable each time, while not using a heuristic function to choose the next value. For each instance, 12 executions have been made, 4 for each algorithm and an average has been exported. For the algorithms taking too long to finish, only 1 complete execution has been made, while the other 3 were interrupted after 60 minutes. For the MIN-CONFLICTS algorithm, the maximum steps were defined to 100000.

In general, MAC-AC3 algorithm does pretty well, with the worst average time being around 2 and a half minutes.<br>
FC algorithm needs on average more time to solve the problems, while for some of them the order of magnitude is increased to hours.<br>
Some deviations are observed due to pseudorandomness used in some functions, such as argmin_random_tie().<br>
MIN-CONFLICTS algorithm is found to be unsuitable for this problem, as it found a solution only for 1 out of the 23 satisfiable problems, and for none of the unsatisfiable problems.

The full statistics of the comparison are freely available to anyone interested, after further contact.
