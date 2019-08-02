function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

z = X * theta;
h_x = sigmoid(z)

reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);

J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))) + reg_term; %regularized cost function

grad(1) = (1/m)* (X(:,1)'*(h_x-y));   % 1 x 1
grad(2:end) = (1/m)*(X(:,2:end)'*(h_x-y)) + (lambda/m)*theta(2:end); % n x 1

end
