function p = predict(theta, X)
%   predict computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

% initialize variables
m = size(X, 1); % Number of training examples
p = zeros(m, 1);

p = round(sigmoid(X * theta));

end
