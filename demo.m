
load('dermatology_uni.mat')



data =fea;  % data is n x d
label =gnd; % label is n x 1

%Normalize data
normA = data - min(data(:));
data = normA ./(max(data(:))-min(data(:))); %;

%-------PCA reudce dimensionality to keep 95% energy
options=[];
options.PCARatio=0.95;
[eigvector, ~] = PCA1(data, options);
data = data*eigvector;

Train_data =data'; %Train_data is a (d x n) matrix.





%%%set parameters
n_class = length(unique(label));
lambda = 0.05;
gamma  = 0.05;
feature_num = 10;



[obj,U,M,W]=RobustL2FKM(Train_data,gamma,lambda,n_class,feature_num);
plot(obj)


                
