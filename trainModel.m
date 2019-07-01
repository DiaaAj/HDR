function [theta] = trainModel(X, y, lambda, num_labels)
%TRAINMODEL Summary of this function goes here
%   Detailed explanation goes here

[m, n] = size(X);
initial_theta = zeros(n+1, 1);
theta = zeros(num_labels, n+1);

options = optimset('GradObj', 'on', 'MaxIter', 50);
for i = 1:num_labels
    fprintf('\nIteration %d of %d\n', i, num_labels);
    if i == 10 
        label = 0; 
    else
        label = i;
    end
    theta(i, :) = ... 
               fmincg(@(t)computeCost(X, (y == label), t, lambda), ...
                                              initial_theta, options);
                                          
end

