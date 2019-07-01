function pred = predict(theta, X)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
m = size(X, 1);
pred = zeros(m, 1);

for i = 1:m
    predict = theta*X(i, :)'; 
    pred(i) = find(predict == max(predict));

end

