function [Wout,Vout, trainingError, testError ] = trainMultiLayer(Xtraining,Dtraining,Xtest,Dtest, W0, V0,numIterations, learningRate )
%TRAINMULTILAYER Trains the network (Learning)
%   Inputs:
%               X* - Training/test features (matrix)
%               D* - Training/test desired output of net (matrix)
%               V0 - Weights of the output neurons (matrix)
%               W0 - Weights of the output neurons (matrix)
%               numIterations - Number of learning setps (scalar)
%               learningRate - The learningrate (scalar)
%
%   Output:
%               Wout - Weights after training (matrix)
%               Vout - Weights after training (matrix)
%               trainingError - The training error for each iteration
%                               (vector)
%               testError - The test error for each iteration
%                               (vector)

% Initiate variables
trainingError = nan(numIterations+1,1);
testError = nan(numIterations+1,1);
numTraining = size(Xtraining,2);
numTest = size(Xtest,2);
numClasses = size(Dtraining,1) - 1;
Wout = W0;
Vout = V0;

estimated_time = -1;
time_elapsed = 0;

% Calculate initial error
Ytraining = runMultiLayer(Xtraining, W0, V0);
Ytest = runMultiLayer(Xtest, W0, V0);
trainingError(1) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
testError(1) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);

tic

for n = 1:numIterations

    [Ytraining,~,U] = runMultiLayer(Xtraining, Wout, Vout);
    
    N = numTraining;%size(W0,1) * size(W0,2) * size(V0,1) * size(V0,2);

    dEdY = 2/N * (Ytraining - Dtraining);
    dYdU = Vout(:,2:end)'; %skip first row with bias, since this does not affect W
    dUdS = tanhprim(U(2:end, :));%1 - U(2:end,:).^2;
    dSdW = Xtraining';
    
    grad_v =  dEdY* U'; %Calculate the gradient for the output layer,
                                %basically similar to single layer
    
    grad_w = dYdU*dEdY.*dUdS*dSdW;


    Wout = Wout - learningRate * grad_w; %Take the learning step.
    Vout = Vout - learningRate * grad_v; %Take the learning step.

    Ytraining = runMultiLayer(Xtraining, Wout, Vout);
    Ytest = runMultiLayer(Xtest, Wout, Vout);

    trainingError(1+n) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
    testError(1+n) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);
    
    if mod(n, 1000) == 0
        clc;
        disp(['Training iteration ', num2str(n) , ' of ', num2str(numIterations), '. ', num2str(n/numIterations * 100), '% done']);
        % Display some info on the training progress
        time_elapsed = time_elapsed + toc;
        estimated_time = time_elapsed / (n / numIterations);
        disp(['Time elapsed: ', num2str(floor(time_elapsed / 60)), 'm ', num2str(mod(time_elapsed,60)), 's.']);
        disp(['Estimated training time: ', num2str(floor(estimated_time / 60)), 'm ', num2str(mod(estimated_time, 60)), ' s.']);
        tic;
    end
end

end
