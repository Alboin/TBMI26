function [ Y, L, U ] = runMultiLayer( X, W, V )
%RUNMULTILAYER Calculates output and labels of the net
%   Inputs:
%               X  - Features to be classified (matrix)
%               W  - Weights of the hidden neurons (matrix)
%               V  - Weights of the output neurons (matrix)
%
%   Output:
%               Y = Output for each feature, (matrix)
%               L = The resulting label of each feature, (vector) 

S = W*X; %Calculate the summation of the weights and the input signals (hidden neuron)
U = tanh(S); %Calculate the activation function as a hyperbolic tangent
U = cat(1,ones(1,length(U)), U);%add bias for output layer
%Y = V'*U; %Calculate the summation of the output neuron
Y = tanh(V * U);   

% Calculate classified labels
[~, L] = max(Y,[],1);
L = L(:);

end

