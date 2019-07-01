function [opt_lambda, min_cost] = ... 
                    computeLambda(X_train, y_train, X_cv, y_cv, num_labels)
%COMPUTELAMBDA finds the value of the optimal lambda 

%lambda testing values 
lambda_values = [
            0.0, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3 ...
            3.5, 4, 5, 6, 8 ...
        ];               

min_cost = inf;
for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    fprintf('\nmodel %f with lambda = %f\n', i, lambda);
    [theta] = trainModel(X_train, y_train, lambda, num_labels);
    cost = computeCost(X_cv, y_cv, theta', num_labels);
    
    if(cost < min_cost)
        min_cost = cost(i);
        opt_lambda = lambda;
    end
    
end
                
end

