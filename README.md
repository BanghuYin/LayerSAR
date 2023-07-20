# LayerSAR: A Efficient and Complete NN Verifier
Abstract—Safety and robustness properties are highly required
for neural networks deployed in safety critical applications.
Current complete verification techniques of these properties
suffer from the lack of efficiency and effectiveness. In this paper,
we present an efficient complete approach to verifying safety and
robustness properties of neural networks through incrementally
determinizing activation states of neurons. The key idea is to
generate constraints via layer-wised splitting that make activation
states of hidden neurons become deterministic efficiently, and
which are then utilized for refining inputs systematically so that
abstract analysis over the refined input can be more precise. Our
approach decomposes a verification problem into a set of subproblems
via layer-wised input space splitting. The property is
then checked in each sub-problem, where the activation states
of at least one hidden neurons will be determinized. Further
checking is accelerated by constraint-guided input refinement.
We have implemented a parallel tool called LayerSAR to verify
safety and robustness properties of ReLU neural networks in a
sound and complete way, and evaluated it extensively on several
benchmark sets. Experimental results show that our approach is
promising, compared with complete tools such as Planet, Neurify,
Marabou, ERAN, Venus, Venus2 and nnenum in verifying safety
and robustness properties on the benchmarks.

# Citing LayerSAR
@article{Yin, Banghu;Chen, Liqian;Liu, Jiangchao;Wang, Ji2022Efficient Complete Verification of Neural Networks via Layerwised Splitting and Refinement,  
title={Efficient Complete Verification of Neural Networks via Layerwised Splitting and Refinement},  
author={Yin, Banghu;Chen, Liqian;Liu, Jiangchao;Wang, Ji},  
journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},  
issue={No.11},  
pages={3898-3909},  
year={2022},  
}


# Main Contributors
Banghu Yin - bhyin@nudt.edu.cn  
Liqian Chen - lqchen@nudt.edu.cn
