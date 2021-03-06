%% This script will help you test out multi single layer neural network code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

clear;
close all;
clc;

dataSetNr = 3; % Change this to load new data

[X, D, L] = loadDataSet( dataSetNr );

% Select a subset of the training features

numBins = 2; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
% XBin1 = Xt{1};
% Modify the X Matrices so that a bias is added

% The Training Data
Xtraining = cat(1,ones(1,length(Xt{1})), Xt{1});

% The Test Data
Xtest = cat(1,ones(1,length(Xt{2})), Xt{2});

minTestVal = min(min(Xtest));
maxTestVal = max(max(Xtest));

minTrainVal = min(min(Xtraining));
maxTrainVal = max(max(Xtraining));

XtestNonNormalized = Xtest;
XtrainingNonNormalized = Xtraining;

if dataSetNr == 4
    %Normalize data if OCR
    Xtest = (Xtest - minTestVal) / ( maxTestVal - minTestVal );
    Xtraining = (Xtraining - minTrainVal) / ( maxTrainVal - minTrainVal );
end




%% Train your multi layer network

              %neurons %iterations %learningrate %weight initialization
trainParameters = [ 10 8000 0.005 0.001; %dataSetNr = 1
                    20 6000 0.005 0.001; %dataSetNr = 2
                    40 20000 0.005 0.001; %dataSetNr = 3
                    80 40000 0.001 0.0001]; %dataSetNr = 4
                
                %the overtraining were performed with parameters:
                % 20 400000 0.005 0.001 and bin size 10 instead of inf

% Note: You need to modify trainMultiLayer() in order to train the network
numLabels = length(unique(Lt{1}));
numHidden = trainParameters(dataSetNr, 1);
numIterations = trainParameters(dataSetNr, 2);
learningRate = trainParameters(dataSetNr, 3);
W0 = rand(numHidden,size(Xtest,1)) * trainParameters(dataSetNr, 4) - (trainParameters(dataSetNr, 4) / 2); % Change this, Initiate your weight matrix W
V0 = rand(numLabels, numHidden+1) * trainParameters(dataSetNr, 4) - (trainParameters(dataSetNr, 4) / 2);

%
tic
[W,V, trainingError, testError ] = trainMultiLayer(Xtraining,Dt{1},Xtest,Dt{2}, W0,V0,numIterations, learningRate );
trainingTime = toc;
% Plot errors
figure(1101)
clf
[mErr, mErrInd] = min(testError);
plot(trainingError,'k','linewidth',1.5)
hold on
plot(testError,'r','linewidth',1.5)
plot(mErrInd,mErr,'bo','linewidth',1.5)
hold off
title('Training and Test Errors, Multi-Layer')
legend('Training Error','Test Error','Min Test Error')

% Calculate The Confusion Matrix and the Accuracy of the Evaluation Data
% Note: you have to modify the calcConfusionMatrix() function yourselfs.

[ Y, LMultiLayerTraining ] = runMultiLayer(Xtraining, W, V);
tic
[ Y, LMultiLayerTest ] = runMultiLayer(Xtest, W,V);
classificationTime = toc/length(Xtest);
% The confucionMatrix
cM = calcConfusionMatrix( LMultiLayerTest, Lt{2})

% The accuracy
acc = calcAccuracy(cM);

display(['Time spent training: ' num2str(trainingTime) ' sec'])
display(['Time spent calssifying 1 feature vector: ' num2str(classificationTime) ' sec'])
display(['Accuracy: ' num2str(acc)])

% Plot classifications
% Note: You do not need to change this code.

if dataSetNr < 4
    plotResultMultiLayer(W,V,XtrainingNonNormalized,Lt{1},LMultiLayerTraining,XtestNonNormalized,Lt{2},LMultiLayerTest)
else
    plotResultsOCR( Xtest, Lt{2}, LMultiLayerTest )
end
