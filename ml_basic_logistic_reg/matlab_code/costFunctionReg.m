function [J, grad] = costFunctionReg(theta, X, y, lambda)
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize variables
m = length(y); % number of training examples
h = sigmoid(X * theta);
J = 0;
grad = zeros(size(theta));
control = ones(size(theta));
control(1) = 0;

cost = sum(((-y .* log(h)) - ((1 - y) .* log(1 - h))))/m;
reg = (lambda/(2 * m)) * sum((theta .* control) .^ 2);

J = cost + reg

reg_g = (lambda / m) * (theta .* control);
for l = 1:columns(X);
    grad = (1/m .* X' * (h - y)) + reg_g;

end
end
