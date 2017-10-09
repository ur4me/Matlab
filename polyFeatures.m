% calculate cost function
diff = X*theta - y;
% calculate penalty
% excluded the first theta value
theta1 = [0 ; theta(2:end, :)];
p = lambda*(theta1'*theta1);
J = (diff'*diff)/(2*m) + p/(2*m);

% calculate grads
grad = (X'*diff+lambda*theta1)/m;

for i= 1:m
    theta          = trainLinearReg(X(1:i,:), y(1:i), lambda);
    error_train(i) = linearRegCostFunction(X(1:i,:)   , y(1:i)   , theta, 0);
    error_val(i)   = linearRegCostFunction(Xval, yval, theta, 0);
end

X_poly(:,1) = X;
for i=2:p
    X_poly(:,i) = X.*X_poly(:,i-1);
end

for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    theta = trainLinearReg(X,y,lambda);
    error_train(i) = linearRegCostFunction(X   , y   , theta, 0);
    error_val(i)   = linearRegCostFunction(Xval, yval, theta, 0);
end
