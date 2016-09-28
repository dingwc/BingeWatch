function [T, lower_bound] = E_updates(v, h, X, pi_inter, beta_inter)
% E-step updates
% Inputs:
% v - N*1 counts
% h - N*1 censorship thresholds
% X - N*D co-variates matrix
% pi_inter - 1*K weights estimates from last step
% beta_inter - D*K coeffecient parameters
%
% Output:
% T - N * K conditional probabilities P(Z|V, last parameters)

% indicator of censorship
c = double(v>=h);
K = numel(pi_inter);
[N,D]=size(X);
% calculating beta
lambdas = exp(X * beta_inter);

% intermediate results
% Probs = lambdas.^repmat(v,1,K).^exp(-lambdas);
Probs = poisspdf(repmat(v,1,K), lambdas);
% add pdf at h to get upper side with P(v>=h)
Ps_upper = poisscdf(repmat(h,1,K), lambdas, 'upper')+poisspdf(repmat(h,1,K), lambdas);
T = Probs.*(1-repmat(c,1,K)) + Ps_upper.*repmat(c,1,K);

%T(c,:) = Ps_upper;
%T(c_comp,:) = Probs;

% re-weights by pi_last
T = repmat(pi_inter, N,1).* T;
lower_bound = sum(log(sum(T,2)));
% normalize for each data point
T = T./repmat(sum(T,2), 1, K);
% T = repmat(pi_inter', N,1).*(Probs.^(1-repmat(c,1,K)))
% .*(Ps_upper.^repmat(c,1,K));
end