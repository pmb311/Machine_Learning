function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some variables
m = length(y); % number of training examples
h = sigmoid(X * theta); % hypothesis
J = 0;

% Calculate J and the gradient
J = sum(((-y .* log(h)) - ((1 - y) .* log(1 - h))))/m

for l = 1:columns(X);
  
  grad = 1/m .* X' * (h - y);

end
end
