function [Wout, trainingError, testError ] = trainSingleLayer(Xt,Dt,Xtest,Dtest, W0,numIterations, learningRate )
%TRAINSINGLELAYER Trains the network (Learning)
%   Inputs:
%               X* - Trainin/test features (matrix)
%               D* - Training/test desired output of net (matrix)
%               W0 - Weights of the neurons (matrix)
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
Nt = size(Xt,2);
Ntest = size(Xtest,2);
Wout = W0;

% Calculate initial error
Yt = runSingleLayer(Xt, W0);
Ytest = runSingleLayer(Xtest, W0);
trainingError(1) = sum(sum((Yt - Dt).^2))/Nt;
testError(1) = sum(sum((Ytest - Dtest).^2))/Ntest;
%    Xij = cat(1, Xt(1,:), Xt(2,:), Xt(1,:), Xt(2,:), Xt(1,:), Xt(2,:));

for n = 1:numIterations
    
    Y = Wout*Xt;
        
    %gradient of the weights
    grad_w = 2*(Y - Dt) * Xt'; 
    
    %gradient descent
    Wout = Wout - learningRate*grad_w;
    trainingError(1+n) = sum(sum((Wout*Xt - Dt).^2))/Nt;
    testError(1+n) = sum(sum((Wout*Xtest - Dtest).^2))/Ntest;
end
end

