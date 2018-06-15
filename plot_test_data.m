clear all;
close all;

plot_output = true;
use_singular_vectors = plot_output;

if plot_output
    X = load('saved_data/test_output_data.txt');
else
    X = load('saved_data/test_input_data.txt');
end

if use_singular_vectors
    [X, ~, ~] = svd(X, 'econ');
    X = X ./ repmat(sqrt(sum(X.^2,2)),1,size(X,2));
end

labels = load('saved_data/test_labels.txt');
X1 = X(labels == 1,:);
X2 = X(labels == 2,:);
X3 = X(labels == 3,:);
plot3(X1(:,1), X1(:,2), X1(:,3), 'bx', X2(:,1), X2(:,2), X2(:,3), 'ro', X3(:,1), X3(:,2), X3(:,3), 'gs')

axis equal
