function [mu sigma2] = estimateGaussian(X)


% Useful variables
[m, n] = size(X);


mu = zeros(n, 1);
sigma2 = zeros(n, 1);



mu = mean(X)';
sigma2 = var(X, 1)';

function [bestEpsilon bestF1] = selectThreshold(yval, pval)


bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    


    tp = sum((yval==1) & (pval<epsilon));

    % yval says it's not an anomaly,  but algorithm says anomaly.
    fp = sum((yval==0) & (pval<epsilon));

    % yval says it's an anomaly,  but algorithm says not anomaly.
    fn = sum((yval==1) & (pval>=epsilon));

    % precision and recall
    prec = tp/(tp+fp);
    rec = tp/(tp+fn);

    % F1 value;
    F1 = (2*prec*rec)/(prec+rec);