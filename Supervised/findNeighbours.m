function [nearestNeighbours] = findNeighbours(traindata, sample, k)

% A vector containing the distances between the sample and each trainging
% sample.
distances = zeros(1,length(traindata));

for i = 1:length(traindata)
    % Calculate the euclidian distance between the sample and each
    % traindata.
    % distances(i) = sqrt(traindata(:,i)' * sample);
    distances(i) = norm(traindata(:,i) - sample);
end

% A vector containing the indices to the k nearest neighbours.
nearestNeighbours = ones(1,k) * -1;

for i = 1:k
    % Find the nearest neighbour
    [~, idx] = min(distances);
    % Remove the choosen neigbour from the list
    distances(idx) = [];
    % Add the index to our list with nearest neighbours
    nearestNeighbours(i) = idx;
end


end

