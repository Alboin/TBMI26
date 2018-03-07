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

% Calculate initial error
Ytraining = runMultiLayer(Xtraining, W0, V0);
Ytest = runMultiLayer(Xtest, W0, V0);
trainingError(1) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
testError(1) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);

for n = 1:numIterations
    [Ytraining,~,U] = runMultiLayer(Xtraining, Wout, Vout);

    dEdY = 2*(Ytraining - Dtraining);
    dYdU = Vout(2:end,:)'; %skip first row with bias, since this does not affect W
    dUdS = 1 - U(2:end,:).^2;
    dSdW = Xtraining';
    
    grad_v =  dEdY* U'; %Calculate the gradient for the output layer,
                                %basically similar to single layer
    %Debugging
    size(dYdU)
    size(dEdY)
    size(dUdS)
    size(dSdW)
    
    size(dUdS * dSdW)
    size(dYdU' * dEdY)
    
    grad_w = (dUdS * dSdW)' .* (dYdU' *dEdY);% * dYdU * dUdS * dSdW ;%..and for the hidden layer.
                    % Here we need to consider the chain rule

    Wout = Wout - learningRate * grad_w; %Take the learning step.
    Vout = Vout - learningRate * grad_v; %Take the learning step.

    Ytraining = runMultiLayer(Xtraining, Wout, Vout);
    Ytest = runMultiLayer(Xtest, Wout, Vout);

    trainingError(1+n) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
    testError(1+n) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);
end

end
