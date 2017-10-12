function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)


% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));




% compute collaborative filtering cost function
temp = ((X * Theta') - Y);
J = 1/2 * sum(sum(R .* (temp.^2)));

% add regularized gradients terms
reg_term_X_grad = lambda * X;
reg_term_Theta_grad = lambda * Theta;
		
% compute collaborative filtering gradients
X_grad = ((X * Theta' - Y) .* R) * Theta;
X_grad = X_grad + reg_term_X_grad;

Theta_grad = ((X * Theta' - Y) .* R)' * X;
Theta_grad = Theta_grad + reg_term_Theta_grad;

% compute regularized cost function
reg_term_J = (lambda/2) * (sum(sum(Theta.^2)) + sum(sum(X.^2))); 
J = J + reg_term_J;










grad = [X_grad(:); Theta_grad(:)];

end
