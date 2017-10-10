function [C, sigma] = dataset3Params(X, y, Xval, yval)

C = 1;
sigma = 0.3;


possible_values_C = [0.01 0.03 0.1 0.3 1 3 10 30];
possible_values_sigma = [0.01 0.03 0.1 0.3 1 3 10 30];

pred_error = 1000;

for Ci=1:length(possible_values_C),
    local_C = possible_values_C(Ci);
    for sigmai=1:length(possible_values_sigma),
        local_sigma = possible_values_sigma(sigmai);
        model = svmTrain(X, y, local_C, @(x1, x2) gaussianKernel(x1, x2, local_sigma));
        predictions = svmPredict(model, Xval);
        local_pred_error = mean(double(predictions ~= yval));
        if local_pred_error < pred_error,
            fprintf('\nPrediction error is %f with C %f and sigma %f', local_pred_error, local_C, local_sigma);
            pred_error = local_pred_error;
            C = local_C;
            sigma = local_sigma;
        end
    end
end

