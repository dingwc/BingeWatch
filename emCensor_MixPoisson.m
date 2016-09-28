% This code is for learning the Mixture Poisson regression Model with 
% Censored Observation in the following Paper:
%
% W. Trouleau, A. Ashkan, W. Ding, and B. Eriksson, 
% “Just One More: Modeling Binge Watching Behavior”, 
% in Proc. ACM International Conference on Knowledge 
% Discovery and Data Mining (SIGKDD), 
% San Francisco, CA, USA, Aug. 13- 17, 2016
%
%
% Code provided by W.Ding 
% Permission is granted for anyone to copy, use, modify, or distribute 
% this program and accompanying programs and documents for any purpose,
% provided this copyright notice is retained and prominently displayed,
% along with a note saying that the original programs are available from 
% our web page. The programs and documents are distributed without any 
% warranty, express or implied. As the programs were written for research 
% purposes only, they have not been tested to the degree that would be 
% advisable in any important application. 
%
% All use of these programs is entirely at the user's own risk.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function [pi,beta, ELBO, sum_ll] = emCensor_MixPoisson(v, h, X, pi_0, beta_0, varargin)
% This is the overall function of learning model parameters of the 
% "Censored Poisson Regression Model with Latent Factors"
% The function take input as the observation dataset and output learned
% model parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% v - N*1 counts,
% h - N*1 censorship thresholds
% X - N*D co-variates matrix
% pi_0 - 1*K initial mixture weights, 
% beta_0 - D*K co-variates coefficients matrix
% (we denote by K the number of mixtures/latent variables)
%
% Outputs:
% pi - 1*K estimated mixtrue weights
% beta - D*K estimated beta coefficients for K Poisson regressions
% ELBO - lower bound on the Log-likelihood
% sum_ll - the overall log-likelihood
% varargin - where you can set your hyper-parameters, includeing:
% -- 'max_iter', the maximum number of iterations in the EM algorithm
% -- 'learn_rate', the gradient descent step-size in the numerical
% optimization to learn \beta in M-step
% -- 'stop_threshold', the stopping cretiria for EM-algorithm, in % of
% changes in ELBO
 

pnames = {'max_iter','learn_rate','stop_threshold'};
dflts =  { [],         [], []};

[max_iter, learn_rate, stopping] = internal.stats.parseArgs(pnames, dflts, varargin{:});

if isempty(max_iter)
    max_iter = 50;
end
if isempty(learn_rate)
    learn_rate = 0.0001;
end
if isempty(stopping)
    stopping = 1e-4;
end


% get dimensions 
[D,K] = size(beta_0);  [N,D] = size(X);
% initialize iterating parameters
pi_inter = pi_0;  
beta_inter = beta_0;

% start EM interations
ELBO = zeros(max_iter, 1);
for i = 1:max_iter
    % E-step : calculate tau's
    [T,ELBO(i)] = E_updates(v, h, X, pi_inter, beta_inter);
    % M-step : updates model parameters 
    [pi_inter,beta_inter] = M_updates(T, v, h, X, beta_inter, learn_rate);
    fprintf('   Iter %d :, likelihood lowerbounds: %f. \n', i, ELBO(i));
    if i>=2
        if abs(ELBO(i)-ELBO(i-1))<=stopping*abs(ELBO(i-1))
                pi = pi_inter;
                beta = beta_inter;
                [~,sum_ll] = E_updates(v, h, X, pi, beta);
                fprintf('The final ll is %f. \n', sum_ll);
                return;
        end
    end
end

% return
pi = pi_inter;
beta = beta_inter;
[~,sum_ll] = E_updates(v, h, X, pi, beta);
fprintf('The final ll is %f. \n', sum_ll);
end