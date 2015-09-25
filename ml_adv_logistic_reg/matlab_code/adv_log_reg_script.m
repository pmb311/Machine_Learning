%     Uses the following files
%	  displayData.m
%	  fmincg.m
%     lrCostFunction.m (logistic regression cost function)
%	  sigmoid.m
%     oneVsAll.m
%     predictOneVsAll.m

%% Initialization
clear ; close all; clc

num_labels = 10;          % 10 labels, from 1 to 10. 0 is mapped to label 10.

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('log_reg_data3.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Vectorized Logistic Regression ============

fprintf('\nSingle iteration of cost function...\n')
lambda = 0.1;
X = [ones(m, 1) X];
n = size(X, 2)
test_theta = zeros(n, 1);

[J, grad] = lrCostFunction(test_theta, X, y, lambda)

fprintf('\nTraining One-vs-All Logistic Regression...\n')


% [all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================
% Compute accuracy on our training set
% pred = predictOneVsAll(all_theta, X);

% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

