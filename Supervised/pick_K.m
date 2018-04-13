%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data
clear;
clc;
dataSetNr = 4; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
% plotCase(X,D)


%% Select a subset of the training features
kmax = 30;
numBins = 4; % Number of Bins you want to divide your data into
numSamplesPerLabelPerBin = 100; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
% XBin2 = Xt{2};

acc = zeros(kmax,1);

%loop through different k's and save the resulting accuracies
for k = 1:kmax
    singleAcc = zeros(numBins,1);
    for bin = 1:numBins

        LkNN = kNN(Xt{mod(bin,numBins)+1}, k, Xt{bin}, Lt{bin});

        % The confusionMatrix
        cM = calcConfusionMatrix( LkNN, Lt{2});

        % The accuracy
        singleAcc(bin) = calcAccuracy(cM);
    end
    acc(k) = sum(singleAcc) / numBins;
end



%show plot over the accuracy for different k's
close all;
plot(1:kmax, acc);
hold on;
[maxAcc, maxInd] = max(acc);
scatter(maxInd, maxAcc);
maxAcc
