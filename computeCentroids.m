function centroids = computeCentroids(X, idx, K)


% Useful variables
[m n] = size(X);

centroids = zeros(K, n);



for k = 1:K
  num_k = 0;
  sum = zeros(n, 1);
  for i = 1:m
    if ( idx(i) == k )
      sum = sum + X(i, :)';
      num_k = num_k + 1;
    end
  end
  centroids(k, :) = (sum/num_k)';
end






