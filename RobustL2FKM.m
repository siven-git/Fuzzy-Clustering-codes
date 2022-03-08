function[OBJ,U,M,W]=RobustL2FKM(X,gamma,lambda,nClass,d)
% X is the data matrix (D x N)
% gamma and lambda are regularization parameters
% nClass is the number of classes
% d is the reduced dimensionality


[D,N] = size(X);


%%%% Initalize -----
U = initfcm(nClass, N);
U = U';  % final N * nClass 
options = [];
options.ReducedDim = d;
[W,~] = PCA1(X',options);
% W = ones(D,d);
%%%% Initalize -----


[~, M, ~] = stepfcm(X', U', nClass, 1.2);
M = M';  % final D * nClass
M = W'*M;
St  =  X*X';
OBJ = zeros(1,60);


for it =1:60
    [obj,M,W]= Reweighted(X,lambda,nClass,M,U,W,d,St);
    OBJ(it) = obj(length(obj))+gamma*trace(U*U');
    if it >2
        if abs(OBJ(it)-OBJ(it-1))<0.0001
            break
        end
    end
    
    %%% update U
    dis = pdist2( X'*W, M', 'euclidean' );
    U = update_robustL2_U(W'*X,M,gamma,dis);
    U = U';
end
OBJ = OBJ(1:it);
