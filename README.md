# BingeWatch

We provide all the code for learning and inferencing the Censored Poisson regression Model with Latent Variables proposed in the following Paper:

W. Trouleau, A. Ashkan, W. Ding, and B. Eriksson, “Just One More: Modeling Binge Watching Behavior”, in Proc. ACM International Conference on Knowledge Discovery and Data Mining (KDD'16), San Francisco, CA, USA, Aug. 13- 17, 2016

The code were used to create all the results in modeling TV watching behavior and discoverying Binge Watching patterns from a Large Video-on-Demond dataset. 

* Permission is granted for anyone to copy, use, modify, or distribute this program and accompanying programs and documents for any purpose,  provided this copyright notice is retained and prominently displayed. The programs and documents are distributed without any warranty, express or implied. As the programs were written for research purposes only, they have not been tested to the degree that would be  advisable in any important application. All use of these programs is entirely at the user's own risk.

The main function is emCensor_MixPoisson.m. Please check the description in this function. In general, your data should have an observed variable (denoted by v), a set of co-variates in regression (denoted by x), and censorship thresholds (denoted by h) for each section. Please check out our paper for further description in guidlines of hyper-parameter settings. 
