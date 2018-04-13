function [ labelsOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels



%Test data
%X = Xt{2};
%k = 4;
%Xt = Xt{1};
%Lt = Lt{1};


labelsOut  = zeros(size(X,2),1);
classes = unique(Lt);
numClasses = length(classes);



% Compare each sample in X with all samples in Xt
% Select the k closest neighbours in Xt, save the indices of these.
% knnIndices is sorted so that the closest neighbour is the first column etc.
%knnIndices = knnsearch(Xt',X', 'K', k, 'IncludeTies', true);

knnIndices = zeros(length(X), k);

for sample = 1:length(X)
    knnIndices(sample,:) = findNeighbours(Xt, X(:,sample), k);
end

for sampleIndex = 1:length(X)
    
    loopAgain = true;
    wasTie = false;
    k_temp = k;
    
    while loopAgain
        %classCounter = zeros(length(numLabels));
        occurrenceMatrix = [classes'; zeros(1,length(classes))];

        %occurrenceMatrix example =
        %| 1 2 3 | first row is the classes
        %| 2 5 2 | second row is number of occurrences of each class

        %Loop over the nearest neighbours of a point.
        for neighbourIndex = 1:k_temp
            %Loop over the number of classes.
            for labelIndex = 1:numClasses
                %Count the number of occurrences of each class.
                %Also, add the distance of each sample of same class.
                if Lt(knnIndices(sampleIndex, neighbourIndex)) == occurrenceMatrix(1,labelIndex)
                    occurrenceMatrix(2,labelIndex) = occurrenceMatrix(2,labelIndex) + 1;
                end
            end
        end

        %Sort the number of occurencies for each class in descending order.
        sortedOccurrences = sort(occurrenceMatrix(2,:), 'descend');
        %If there exist a tie.
        if sortedOccurrences(1) == sortedOccurrences(2)
            
            %Debugging
            %disp('Tie!')
            %Lt(knnIndices{sampleIndex}(1:k_temp))
            
            k_temp = k_temp - 1;
            %remove the final element, by changing the class index to a neg value. 
            knnIndices(sampleIndex, k_temp + 1) = k_temp - k;

            %run function again
            loopAgain = true;
            wasTie = true;
        else
            %there is no tie
            loopAgain = false;

            % Classify X using Lt to look up the classes of closest neighbours in Xt.
            % ('mode()' selects the most frequent value in a vector)
            % use only the k_temp closest neighbours (no ties).
            labelsOut(sampleIndex) = mode(Lt(knnIndices(sampleIndex, 1:k_temp)));
            
            % Debugging
            %if wasTie
                %Lt(knnIndices{sampleIndex}(1:k_temp))
            %end
        end
        if wasTie
            disp('A tie occured!');
        end
    end
end




% Classify X using Lt to look up the classes of closest neighbours in Xt.
% ('mode()' selects the most frequent value in a vector)
%for i = 1:length(X)
%    labelsOut(i) = mode(Lt(knnIndices(i,:)));
%end

end

