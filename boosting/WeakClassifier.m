function C = WeakClassifier(T, P, X)
% WEAKCLASSIFIER Classify images using a decision stump.
% Takes a vector X of scalars obtained by applying one Haar feature to all
% training images. Classifies the examples using a decision stump with
% cut-off T and parity P. Returns a vector C of classifications for all
% examples in X.


%Create a vector of 1's.
C = ones(length(X),1) * P;
%Set the the ones that are below the threshold T to -1.
C(X < T) = -1 * P;


%C = (X * P < T * P) * -1;


% You are not allowed to use a loop in this function.
% This is for your own benefit, since a loop will be too slow to use
% with a reasonable amount of Haar features and training images.
    
end

