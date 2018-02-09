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
nbrTrainExamples = 1000;
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];

% Set number of test examples
nbrTestExamples = 3000;

testImages  = cat(3,faces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)),...
                    nonfaces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];


% Loop variables
accuracies_test = 0;
accuracies_train = 0;
numberOfClassifiers = 0;
total_weak_classifiers = maxNumberClassifiers + floor(maxNumberClassifiers/2) * maxNumberClassifiers - (1 - mod(maxNumberClassifiers,2)) * (maxNumberClassifiers / 2);

time_elapsed = 0;
estimated_time = 0;
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
    

    final_classes_test = zeros(length(yTest), 1);
    final_classes_train = zeros(length(yTrain), 1);
    

    % Evaluate on test and training data
    for classifier = 1:nClassifiers
        
        haar_idx = weak_classifiers(classifier, 1);
        threshold = weak_classifiers(classifier, 2);
        polarity = weak_classifiers(classifier, 3);
        error = weak_classifiers(classifier, 4);

        % Classify all images with one weak classifier
        C_test = WeakClassifier(threshold, polarity, xTest(haar_idx,:));
        C_train = WeakClassifier(threshold, polarity, xTrain(haar_idx,:));
        
        alpha = 0.5 * log((1 - error)/error);
        
        % Sum the results from the weak classifiers
        final_classes_test = final_classes_test + alpha * C_test;
        final_classes_train = final_classes_train + alpha * C_train;
    end

    final_strong_classifier_test = sign(final_classes_test);
    final_strong_classifier_train = sign(final_classes_train);
    
    accuracy_test = sum(final_strong_classifier_test == yTest')/length(final_strong_classifier_test);
    accuracy_training = sum(final_strong_classifier_train == yTrain')/length(final_strong_classifier_train);
    
    % Save the accuracy for this strong classifier and the number of weak
    % classifiers that it consists of.
    accuracies_test = [accuracies_test, accuracy_test];
    accuracies_train = [accuracies_train, accuracy_training];
    
    numberOfClassifiers = [numberOfClassifiers, nClassifiers];
    % Also save the configurations for the strong classifier.
    strong_classifiers(:,:, nClassifiers) = weak_classifiers;
    
    % Display some info on the training progress
    clc;
    time_elapsed = time_elapsed + toc;
    if estimated_time < 0.1
        estimated_time = time_elapsed * total_weak_classifiers;
    end
    disp(['Training strong classifier ',num2str(nClassifiers), ' of ' num2str(maxNumberClassifiers), '. ']);
    disp(['Using ', num2str(nClassifiers), ' weak classifiers.']);
    disp([num2str((nClassifiers / maxNumberClassifiers) * 100), '% Done.']);
    disp(['Time elapsed: ', num2str(floor(time_elapsed / 60)), 'm ', num2str(mod(time_elapsed,60)), 's.']);
    disp(['Estimated training time: ', num2str(floor(estimated_time / 60)), 'm ', num2str(mod(estimated_time, 60)), ' s.']);
    tic;

end

% Remove first dummy-element.
accuracies_test = accuracies_test(2:end);
accuracies_train = accuracies_train(2:end);
numberOfClassifiers = numberOfClassifiers(2:end);

% Plot the accuracy as a function of how many weak classifiers in each
% strong classifier.
plot(numberOfClassifiers, accuracies_train, '--', numberOfClassifiers, accuracies_test, [1, maxNumberClassifiers], [0.8, 0.8], ':');
legend('Accuracy on training samples','Accuracy on test samples','Location','southeast')
xlabel('Number of weak classifiers');
ylabel('Accuracy');
ylim([0.0 1.0]);
title('Strong classifier accuracies');

[best_accuracy_test, best_classifier_index] = max(accuracies_test);

% Print some results and used parameters.
fprintf('\n');
disp(['Best test-data accuracy (', num2str(best_accuracy_test), ') is given with ', num2str(best_classifier_index), ' weak classifiers.']);
disp(['Number of training samples used: ', num2str(nbrTrainExamples)]);
disp(['Number of test samples used: ', num2str(nbrTestExamples)]);
disp(['Number of Haar-filters generated: ', num2str(nbrHaarFeatures)]);

% Plot Haar-filters used by the best classifier.
figure(2);
colormap gray;
for k = 1:maxNumberClassifiers
    % Check if it is a weak classifier or just empty data
    if strong_classifiers(k,4,best_classifier_index) < 0.9999
        haar_index = strong_classifiers(k,1, best_classifier_index);
        subplot(5,5,k),imagesc(haarFeatureMasks(:,:,haar_index),[-1 2]);
        axis image;
        axis off;
    end
end


% Extract some faces/non-faces that were missclassified  by the best
% classifier.

% Evaluate on test data
for k = 1:maxNumberClassifiers
    % Check if it is a weak classifier or just empty data
    if strong_classifiers(k,4,best_classifier_index) < 0.9999
        haar_idx = strong_classifiers(k, 1, best_classifier_index);
        threshold = strong_classifiers(k, 2, best_classifier_index);
        polarity = strong_classifiers(k, 3, best_classifier_index);
        error = strong_classifiers(k, 4, best_classifier_index);

        % Classify test images with one weak classifier
        C_test = WeakClassifier(threshold, polarity, xTest(haar_idx,:));
        
        alpha = 0.5 * log((1 - error)/error);

        % Sum the results from the weak classifiers
        final_classes_test = final_classes_test + alpha * C_test;
    end
end
% Do the classification
final_strong_classifier_test = sign(final_classes_test);

% Get missclassified test-cases
missclassified = final_strong_classifier_test ~= yTest';


figure(3);
colormap gray;
image_number = 1;
% Find the images corresponding to the missclassifications and show 25 of them.
% Show 12 missclassified faces.
for i = 1:length(missclassified)
    if missclassified(i)
        subplot(5,5,image_number), imagesc(faces(:,:,i));
        axis image;
        axis off;
        image_number = image_number + 1;
    end
    if image_number == 13
        break;
    end
end
% Show 13 missclassified non-faces.
for i = length(missclassified)/2:length(missclassified)
    if missclassified(i)
        subplot(5,5,image_number), imagesc(nonfaces(:,:,i));
        axis image;
        axis off;
        image_number = image_number + 1;
    end
    if image_number == 26
        break;
    end
end
