%% Handwritten digits recognition
% description here

%% initialization
clear; close all; clc;

%load mnist data
[X, y] = mnist_parse('mnist_dataset\train-images.idx3-ubyte', ...
                                'mnist_dataset\train-labels.idx1-ubyte');
                            
X = double(reshape(X, size(X,1)*size(X,2), []).');
y = double(y);
                            
[X_test, y_test] = mnist_parse('mnist_dataset\t10k-images.idx3-ubyte', ...
                                'mnist_dataset\t10k-labels.idx1-ubyte');

X_test = double(reshape(X_test, size(X_test,1)*size(X_test,2), []).');
y_test = double(y_test); 
                            
[m, n] = size(X);
num_labels = 10;
% last 20 percent of X for cross validation dataset and 80 percent ...
% for the training dataset 
X_train = X(1:ceil(m*0.8), :); y_train = y(1:ceil(m*0.8), :);
X_cv = X(floor(m*0.8:end), :); y_cv = y(floor(m*0.8:end), :);


%% visualize a sample test of the data
indices = randperm(m);
sel = X(indices(1:100), :);
displayData(sel);

%% optimizing lambda value and training the model
%[lambda, cost] = computeLambda(X_train, y_train, X_cv, y_cv, num_labels);
%fprintf('\nOptimal lambda value = %f\tcost = %f\n', lambda, cost);
%save('lambda'); save('cost');

%load('lambda'); %optimal value 
%theta = trainModel(X_train, y_train, lambda, num_labels);
%save('theta')

%% Test the model and accuracy
load('theta');
pred = predict(theta,[zeros(size(X_test, 1), 1) X_test]);
pred(pred == 10) = 0; 
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

for i = 1:size(X_test, 1)
    p = predict(theta, [0 X_test(i, :)]);
    if p == 10 
        p = 0;
    end
    displayData(X_test(i, :));
    fprintf('\npredicted value = %d\tright value = %d\n', p, y_test(i));
    
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end