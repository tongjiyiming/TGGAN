function [X] = removeNAN(X)
X(any(isnan(X), 2), :) = [];
end