function[OBJ,M,W]=Reweighted(X,lambda,nClass,M,U,W,d,St)
% X is the data matrix (D x N)
% gamma and lambda are regularization parameters
% nClass is the number of classes
% d is the reduced dimensionality



[D,N] = size(X);

iter = 10;
obj = zeros(iter,1);



% update weighted
dis = distfcm( X'*W, M');
P = 0.5./(dis+1e-6);
S = P.*U;
II = 10*eye(D);
for IT =1:iter
    
    for it=1:5
        
        
        %%% update W
        B = zeros(N,N);
        for i=1:N
            B(i,i)=sum(S(i,:));
        end
        C = zeros(nClass,nClass);
        for i=1:nClass
            C(i,i)=1/(sum(S(:,i))+1e-6);
        end
        Sw = X*(B-S*C*S')*X';
        if it ==1
            [~,eigvalue]=eig(Sw);%求矩阵的特征值和特征向量，x为特征向量矩阵，y为特征值矩阵。
            eigenvalue=diag(eigvalue);%求对角线向量
            reg = max(eigenvalue);%求最大特征值
        end
        Sb = lambda*St-Sw+II*real(reg);
        W  = eig1(Sb,d,1);
%         W = real(W);
        a =1;
        %%% update  M
        M = update_PM(W'*X,S');
        TT = X-W*W'*X;
        
        obj(it) = trace(W'*Sw*W)+lambda*trace(TT*TT');
    end
    
    % update weighted
%     dis = pdist2( X'*W, M', 'euclidean' );
    dis = distfcm( X'*W, M');
    P = 0.5./(dis+1e-6);
    S = P.*U;
    
    %%% update W
    B = zeros(N,N);
    for i=1:N
        B(i,i)=sum(S(i,:));
    end
    C = zeros(nClass,nClass);
    for i=1:nClass
        C(i,i)=1/(sum(S(:,i))+1e-6);
    end
    Sw = X*(B-S*C*S')*X';
    TT = X-W*W'*X;
    OBJ(IT) = trace(W'*Sw*W)+lambda*trace(TT*TT');
    if IT >2
        if abs(OBJ(IT)-OBJ(IT-1))<0.0001
            break
        end
    end
end
OBJ = OBJ(1:IT);
