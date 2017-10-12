function [U, S] = pca(X)

[m, n] = size(X);

U = zeros(n);
S = zeros(n);

Sigma = (1/m)*(X'*X);
[U, S, V]=svd(Sigma);


function Z = projectData(X, U, K)


Z = zeros(size(X, 1), K);


U_reduce = U(:, 1:K);
for i = 1: size(X, 1)
  Z(i, :) = (U_reduce'*X(i, :)')';
end


function X_rec = recoverData(Z, U, K)

X_rec = zeros(size(Z, 1), size(U, 1));

    

U_reduce = U(:, 1:K);
for i = 1:size(Z, 1)
  X_rec(i, :) = (U_reduce*Z(i, :)')';
end








