function X = normalize_data(X)
length_basis = size(X,1);
X = X - mean(X,1);
X = X ./ sqrt(sum(X.^2));
X(~isfinite(X)) = 1/sqrt(length_basis);
end