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

%Calculate distances
distances = zeros(size(X,2),size(Xt,2));
for i = 1:size(X,2)
    for j = 1:size(Xt,2)
        distances(i,j)=euclideanDist(X(:,i),Xt(:,j));
    end
end

%Get classes for nearest neighbors
nns = zeros(size(X,2),k);
for i = 1:size(X,2)
    [~,sortIndex] = sort(distances(i,:),'ascend');
    minIndex = sortIndex(1:k);
    nns(i,:) = Lt(minIndex);
end

%Majority vote
labelsOut = mode(nns,2);

%labelsOut = nns;
end

function [ dist ] = euclideanDist(x1, x2)
        dim = length(x1);
        sum = 0;
        for i = 1:dim
            sum = sum+(x1(i)-x2(i))^2;
        end
        dist = sqrt(sum);
end