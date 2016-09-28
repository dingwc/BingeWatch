function [pi_inter,beta_inter] = M_updates(T, v, h, X, beta_init, learn_rate)
% Inputs:
% T - N*K conditional probabilities
% v - N*1 counts
% h - N*1 censorship thresholds
% X - N*D co-variates
%
% Outputs:
% pi_inter - 1*K new weights params. 
% beta_inter - D*K factor coefficients params. 

% update weights pi_k
pi_inter = sum(T,1);
pi_inter = pi_inter/sum(pi_inter);
% update factors beta_k
beta_inter = update_beta(v, h, X, beta_init, T, learn_rate);

end 