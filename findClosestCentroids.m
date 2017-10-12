function idx = findClosestCentroids(X, centroids)


% Set K
K = size(centroids, 1);

idx = zeros(size(X,1), 1);



for i = 1:size(X, 1)
  min_d = inf;
  for k = 1:K
    diff = X(i, :)'-centroids(k, :)';
    d = diff'*diff;
    if (d < min_d)
      idx(i) = k;
      min_d = d;
    end
  end
end

