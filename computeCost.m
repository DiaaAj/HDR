function [J, grad] = computeCost(X, y, theta, lambda)
%COMPUTECOST a vectorized implementation of cost function and gradient 
% descent for logistic regression
[m, ~] = size(X);
X = [ones(m, 1) X];
% dimensions for mnist: X -> Mx784, theta -> 1x785, h -> 1xM,

h = sigmoid(X*theta);

J = -(y'*log(h)+(1-y)'*log(1-h))/m;
J = J + lambda*sum(theta(2:end).^2)/(2*m); 

grad = (((h - y)'*X))'/m + ... 
                    lambda*[zeros(1, size(theta, 2)); theta(2:end, :)]/m;

end

