function [k] = grbf(x1, x2, sigma)
[n, ~] = size(x1);
[m, ~] = size(x2);
k1 = sum(x1.*x1, 2);
k2 = sum(x2.*x2, 2);

h = repmat(k1, [1, m]) + repmat(k2', [n, 1]);

h = h - 2 * x1*x2';
k = exp(-1 * h / 2 / sigma^2);
end

