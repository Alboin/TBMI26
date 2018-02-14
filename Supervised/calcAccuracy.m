function [ acc ] = calcAccuracy( cM )
%CALCACCURACY Takes a confusion matrix amd calculates the accuracy

% Replace with your own code
acc = trace(cM) / sum(sum(cM));

end

