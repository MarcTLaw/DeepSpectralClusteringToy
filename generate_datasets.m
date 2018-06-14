close all;

d = 3; % input space dimensionality
T = 1000; % minimum number of examples per subcluster

nb_classes = 3; % number of ground truth categories
K = nb_classes; % number of desired clusters
noisy = true; % the experiment corresponds to Table 2 if noisy = true, and to Table 3 otherwise

same_cluster_size = false; % if same_cluster_size = true and noisy = true, the experiment corresponds to Table 1
if same_cluster_size
    defined_size_1 = T;
    defined_size_2 = T;
    defined_size_3 = T;
else
    defined_size_1 = T;
    defined_size_2 = 2*T;
    defined_size_3 = 4*T;
end

%%%
% Creation of the training dataset
%%%

u = zeros(2*T*K,K); 
space = 6;
double_space = 10*space;

C1_1 = repmat([-space,0,0],defined_size_1,1); % Creation of the first subcluster of the first category with T observations
C1_2 = repmat([0, double_space,0],defined_size_1,1); % Creation of the second subcluster of the first category with T observations
C1_3 = repmat([space, double_space,space*3],defined_size_1,1); % Creation of the third subcluster of the first category with T observations
C2_1 = repmat([space,0,0],defined_size_2,1); % Creation of the first subcluster of the second category with 2T observations
C2_2 = repmat([space,double_space,0],defined_size_2,1); % Creation of the second subcluster of the second category with 2T observations
C2_3 = repmat([-space,double_space,0],defined_size_2,1); % Creation of the third subcluster of the second category with 2T observations
C3_1 = repmat([space*0.5,double_space,space],defined_size_3,1); % Creation of the first subcluster of the third category with 4T observations
C3_2 = repmat([-space,0,space],defined_size_3,1); % Creation of the second subcluster of the third category with 4T observations
C3_3 = repmat([-space,-space,-space],defined_size_3,1); % Creation of the third subcluster of the third category with 4T observations


if ~noisy
    noisy = 0.5;
end

X1_1 = noisy * randn(defined_size_1,d)+ C1_1; % Inclusion of normally distributed noise
X1_2 = noisy * randn(defined_size_1,d)+ C1_2; % Inclusion of normally distributed noise
X1_3 = noisy * randn(defined_size_1,d)+ C1_3; % Inclusion of normally distributed noise
X2_1 = noisy * randn(defined_size_2,d)+ C2_1; % Inclusion of normally distributed noise
X2_2 = noisy * randn(defined_size_2,d)+ C2_2; % Inclusion of normally distributed noise
X2_3 = noisy * randn(defined_size_2,d)+ C2_3; % Inclusion of normally distributed noise
X3_1 = noisy * randn(defined_size_3,d)+ C3_1; % Inclusion of normally distributed noise
X3_2 = noisy * randn(defined_size_3,d)+ C3_2; % Inclusion of normally distributed noise
X3_3 = noisy * randn(defined_size_3,d)+ C3_3; % Inclusion of normally distributed noise
figure1 = figure;
plot3(X1_1(:,1), X1_1(:,2), X1_1(:,3), 'bx',X1_2(:,1), X1_2(:,2), X1_2(:,3), 'bx',X1_3(:,1), X1_3(:,2), X1_3(:,3), 'bx',X2_1(:,1), X2_1(:,2), X2_1(:,3), 'ro',X2_2(:,1), X2_2(:,2), X2_2(:,3), 'ro',X2_3(:,1), X2_3(:,2), X2_3(:,3), 'ro',X3_1(:,1), X3_1(:,2), X3_1(:,3), 'gs',X3_2(:,1), X3_2(:,2), X3_2(:,3), 'gs',X3_3(:,1), X3_3(:,2), X3_3(:,3), 'gs');
axis equal
set(gca,'FontSize',15)  
title('Training data (one color per cluster)')


size_1 = size(C1_1,1) + size(C1_2,1) + size(C1_3,1);
size_2 = size(C2_1,1) + size(C2_2,1) + size(C2_3,1);
size_3 = size(C3_1,1) + size(C3_2,1) + size(C3_3,1);


X = [X1_1;X1_2;X1_3;X2_1;X2_2;X2_3;X3_1;X3_2;X3_3];

Y = sparse((1:size(X,1)),[ones(size_1,1);repmat(2,size_2,1);repmat(3,size_3,1)],1,size(X,1),K);
Y = full(Y);

rand_indices = randperm(size(X,1));
is_training = rand_indices <= 15000;
X_train = X(is_training,:);
X_test = X(~is_training,:);

Y_train = Y(is_training,:);
Y_test = Y(~is_training,:);

save('saved_data/X_train.txt', 'X_train', '-ascii');
save('saved_data/X_test.txt', 'X_test', '-ascii');
save('saved_data/Y_train.txt', 'Y_train', '-ascii');
save('saved_data/Y_test.txt', 'Y_test', '-ascii');
