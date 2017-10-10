function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));




X = [ones(m, 1) X];
A1 = sigmoid(X * Theta1');

A1x = [ones(size(A1, 1), 1) A1];
H = sigmoid(A1x * Theta2');

Y = [y==1, y==2];
for i=3:max(y),
    Y = [Y, y==i];
end

s = sum(-Y.*log(H) - (1-Y).*log(1-H));

J = sum(s)/m;

% ADD REGULARIZATION FACTOR

% dont regolarize bias unit
addJ = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = J + addJ;

% 3rd exercise: implement backprop - DOES NOT WORK

for t=1:m, 
    Delta3 = H(t,:) - Y(t,:)
    Delta2 = Theta2(t,:)' * Delta3 %.* sigmoidGradient(A1(t,:)')
    Delta2 = Delta3 * Theta2 .* sigmoidGradient(A1x(t, :))
    Delta2 = Delta2(2:end,:)
    Theta1_grad = Theta1_grad + A1'*Delta2
    Theta2_grad = Theta2_grad + Delta3*H'
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end