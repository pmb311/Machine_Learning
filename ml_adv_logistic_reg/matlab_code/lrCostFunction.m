function [J, grad] = lrCostFunction(theta, X, y, lambda)
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize variables
m = length(y); % number of training examples
h = sigmoid(X * theta); % hypothesis
beta = h - y; % helper for vectorized gradient
J = 0; % cost
grad = zeros(size(theta)); % gradient
control = ones(size(theta)); % ones vector for regularization parameter
control(1) = 0; % first element must be zero (don't regularize the element corresponding to X's ones column)

% Compute cost and regularization parameter
cost = sum(((-y .* log(h)) - ((1 - y) .* log(1 - h))))/m;
reg = (lambda/(2 * m)) * sum((theta .* control) .^ 2);

% Regularized cost
J = cost + reg;

% Compute gradient
grad = (1/m) .* (X' * beta);

% Compute regularization parameter for gradient
temp = theta .* control;
reg_g = (lambda / m) * temp;

% Regularized gradient
grad = grad + reg_g;
grad = grad(:);

end
