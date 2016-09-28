function beta_inter = update_beta(v, h, X, beta_init, T, learn_rate)
% update beta parameters in M-step 
% using convex optimization and gradient descent tools
%
% Inputs: 
% v - N*1 counts
% h - N*1 thresholds
% X - N*D co-variates
% beta_init - D*K params.
%
% Outputs:
% beta_inter - D*K updated params. -> optimal maximizor

max_iter = 2000;
beta_inter = beta_init;
%learn_rate = 0.00005;
%
t = 1;
gradient = calculate_gradient_beta_new(v, h , X, beta_inter, T);
while norm(gradient,'fro')>=0.01 && t<=max_iter
    gradient = calculate_gradient_beta_new(v, h , X, beta_inter, T);    
    beta_inter = beta_inter + learn_rate*gradient;
    t = t+1;
    if mod(t,100)==0
        learn_rate = learn_rate*0.95;
    end
end
fprintf('The final gradient norm %f, at step %d' ,norm(gradient,'fro'), t);


% for t = 1:max_iter
%     % gradient - D*K, each column=gradient vector
%     if mod(t,80)==0
%         learn_rate = learn_rate*0.95;
%     end
%     gradient = calculate_gradient_beta(v, h , X, beta_inter, T);
%     if norm(gradient,2)<0.01
%         fprintf('The final gradient norm %f, \n' ,norm(gradient,2));
%         break;
%     end
%     % maximization 
%     beta_inter = beta_inter + learn_rate*gradient;
% end
% 
% fprintf('The final gradient norm %f, ' ,norm(gradient,2));