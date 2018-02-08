% Load face and non-face data and plot a few examples
clc;
close all;
clear;
load faces;
load nonfaces;
faces = double(faces);
nonfaces = double(nonfaces);

%figure(1);
%colormap gray;
%for k=1:25
%    subplot(5,5,k), imagesc(faces(:,:,10*k));
%    axis image;
%    axis off;
%end

%figure(2);
%colormap gray;
%for k=1:25
%    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
%    axis image;
%    axis off;
%end

% Generate Haar feature masks
nbrHaarFeatures = 100;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

%figure(3);
%colormap gray;
%for k = 1:25
%    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
%    axis image;
%    axis off;
%end

% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = 50;
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];


%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

M = nbrTrainExamples * 2; %faces and non-faces
K = nbrHaarFeatures;

% Initial parameters
D = ones(M,1)/M; %weights
errors = ones(K,1) * 10; %minimum errors
polarities = ones(K,1);
thresholds = zeros(K,1); %classification thresholds

for row = 1:K
    
    E_min = 10;
    
    for col = 1:M
        
        T = xTrain(row,col);
        P = 1;
        
        % Decide classes C with T and P
        C = WeakClassifier(T, P, xTrain(row,:));

        % Use C to measure error E
        E = WeakClassifierError(C, D, yTrain);
        
        
        % Change polarity if error > 0.5
        if E > 0.5
            P = -1;
            E = 1 - E;
        end
        
        % If error is smaller than previously smallest, save error,
        % polarity and threshold.
        if E < E_min
            E_min = E;
            polarities(row) = P;
            thresholds(row) = T;
            errors(row) = E_min;
        end
    end
    
    
    % Use the found minumum error to calculate alpha
    alpha = 0.5 * log((1 - E_min)/E_min);
    
    % Use the found best threshold and polarity to get classifications
    C = WeakClassifierError(thresholds(row), polarities(row), xTrain);
    
    % Calculate new weights using the old ones and the number of correct
    % classifications.
    for i = 1:M
        D(i) = D(i) * exp(-alpha * (C(i) == yTrain(i))); 
    end
    
    % Normalize weights
    D = D / sum(D);
    
end    

%% Extract test data

nbrTestExamples = 3000;

testImages  = cat(3,faces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)),...
                    nonfaces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];

%% Evaluate your strong classifier here
%  You can evaluate on the training data if you want, but you CANNOT use
%  this as a performance metric since it is biased. You MUST use the test
%  data to truly evaluate the strong classifier.

final_classes = zeros(length(yTest), 1);

% Evaluate on training data
for classifier = 1:K
    
    % Classify all images with one weak classifier
    classes= WeakClassifier(thresholds(classifier), polarities(classifier), xTest(classifier,:));
    
    alpha = 0.5 * log((1 - errors(classifier))/errors(classifier));
    final_classes = final_classes + alpha * classes;
    
end

final_classes = sign(final_classes);
accuracy = sum(final_classes == yTest')/length(final_classes)


%% Plot the error of the strong classifier as  function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.



