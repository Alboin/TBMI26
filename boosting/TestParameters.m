clear;
close all;
load faces;
load nonfaces;
faces = double(faces);
nonfaces = double(nonfaces);

% The maximum number of weak classifiers in a strong one.
maxNumberClassifiers = 10;

% Set number of Haar features
nbrHaarFeatures = 300;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

% Set number of training examples
nbrTrainExamples = 500;
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];

% Set number of test examples
nbrTestExamples = 4000;

testImages  = cat(3,faces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)),...
                    nonfaces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];


% Loop variables
accuracies = 0;
numberOfClassifiers = 0;

time_elapsed = 0;
tic;

strong_classifiers = zeros(nbrHaarFeatures, 4, maxNumberClassifiers);

% Initial parameters
M = nbrTrainExamples * 2; %faces and non-faces
K = nbrHaarFeatures;
D = ones(1,M)/M; %weights

for nClassifiers = 1:maxNumberClassifiers

    %contains Haar idx, threshold, polarity, min error
    weak_classifiers = ones(nbrHaarFeatures, 4);

    for classifier = 1:nClassifiers
        
        E_min = weak_classifiers(classifier, 4);
        
        for row = 1:K
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
                    weak_classifiers(classifier, 1) = row; %save Haar index
                    weak_classifiers(classifier, 2) = T;
                    weak_classifiers(classifier, 3) = P;
                    weak_classifiers(classifier, 4) = E_min;
                end
            end
        end
                
        haar_idx = weak_classifiers(classifier, 1);
        threshold = weak_classifiers(classifier, 2);
        polarity = weak_classifiers(classifier, 3);
        
        % Use the found minumum error to calculate alpha
        alpha = 0.5 * log((1 - E_min)/E_min);

        % Use the completed weak classifier to classify the images once
        % more.
        C = WeakClassifier(threshold, polarity, xTrain(haar_idx,:));
        
        
        % Calculate new weights using the old ones and the number of correct
        % classifications from our weak classifier.
        D = D .* exp(-alpha * (C' == yTrain)); 

        % Normalize weights
        D = D / sum(D);
        
    end
    

    final_classes = zeros(length(yTest), 1);

    % Evaluate on training data
    for classifier = 1:nClassifiers
        
        haar_idx = weak_classifiers(classifier, 1);
        threshold = weak_classifiers(classifier, 2);
        polarity = weak_classifiers(classifier, 3);
        error = weak_classifiers(classifier, 4);

        % Classify all images with one weak classifier
        C = WeakClassifier(threshold, polarity, xTest(haar_idx,:));
        
        alpha = 0.5 * log((1 - error)/error);
        
        % Sum the results from the weak classifiers
        final_classes = final_classes + alpha * C;

    end

    final_strong_classifier = sign(final_classes);
    
    accuracy = sum(final_strong_classifier == yTest')/length(final_strong_classifier);
    
    % Save the accuracy for this strong classifier and the number of weak
    % classifiers that it consists of.
    accuracies = [accuracies, accuracy];
    numberOfClassifiers = [numberOfClassifiers, nClassifiers];
    % Also save the configurations for the strong classifier.
    strong_classifiers(:,:, nClassifiers) = weak_classifiers;
    
    % Display some info on the training progress
    clc;
    time_elapsed = time_elapsed + toc;
    disp(['Training strong classifier ',num2str(nClassifiers), ' of ' num2str(maxNumberClassifiers), '. ']);
    disp(['Using ', num2str(nClassifiers), ' weak classifiers.']);
    disp([num2str((nClassifiers / maxNumberClassifiers) * 100), '% Done.']);
    disp(['Time elapsed: ', num2str(time_elapsed), ' seconds.']);
    tic;

end

% Remove first dummy-element.
accuracies = accuracies(2:end);
numberOfClassifiers = numberOfClassifiers(2:end);

% Plot the accuracy as a function of how many weak classifiers in each
% strong classifier.
plot(numberOfClassifiers, accuracies, [1, maxNumberClassifiers], [0.8, 0.8], ':');
xlabel('Number of weak classifiers');
ylabel('Accuracy');
ylim([0.0 1.0]);
title('Strong classifier accuracies');

[best_accuracy, best_classifier_index] = max(accuracies);

fprintf('\n');
disp(['Best accuracy (', num2str(best_accuracy), ') is given with ', num2str(best_classifier_index), ' weak classifiers.']);
disp(['Number of training samples used: ', num2str(nbrTrainExamples)]);
disp(['Number of test samples used: ', num2str(nbrTestExamples)]);
disp(['Number of Haar-filters generated: ', num2str(nbrHaarFeatures)]);

% Plot Haar-filters used by the best classifier.
figure(2);
colormap gray;
for k = 1:maxNumberClassifiers
    if strong_classifiers(k,4,best_classifier_index) < 0.9999
        haar_index = strong_classifiers(k,1, best_classifier_index);
        subplot(5,5,k),imagesc(haarFeatureMasks(:,:,haar_index),[-1 2]);
        axis image;
        axis off;
    end
end