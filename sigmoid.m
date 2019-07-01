function [g] = sigmoid(z)
%SIGMOID computes the hypothesis function for logistic regression
g = 1.0./(1.0 + exp(-z));

end

