function gradient = calculate_gradient_beta_new(v, h , X, beta_inter, T)
% calculating gradient w.r.t. beta 
% Inputs:
% v - N*1 counts
% h - N*1 thresholds
% X - N*D co-variates
% beta_init - D*K params.
%
% Outputs:
% gradient - D*K gradients

% get param. dimensions
[N,D] = size(X);
[D,K] = size(beta_inter);
% censorship indicators
c = double(v>=h);
pidx = find(v>=h);

% intermediate results
lambdas = exp(X*beta_inter);
dQ1 = repmat(1-c,1,K).*(repmat(v,1,K)-lambdas);
dQ2 = zeros(size(dQ1));
% in an old version, this is not included 
dQ2_tmp = lambdas(pidx,:).*poisspdf(repmat(h(pidx)-1,1,K), lambdas(pidx,:))./(poisscdf(repmat(h(pidx),1,K),lambdas(pidx,:),'upper')+poisspdf(repmat(h(pidx),1,K),lambdas(pidx,:)));
dQ2(pidx,:)=dQ2_tmp;
% calculate the gradients in matrix form
gradient = X'*((dQ1+dQ2).*T);
end