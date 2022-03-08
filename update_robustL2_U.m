function A = update_robustL2_U(X,B,gama,D)

[~,n] = size(X);
[~,c] = size(B);
A = zeros(c,n);   


%%%% in this papaer:
%U(i,:) ~ -H_ba(i,:)/2*gama  ---> U(i,:)*sqrt(2*gama) ~ -H_ba(i,:)/sqrt(2*gama) 

for i = 1:n
    dnew = - 0.5.*D(i,:)'/(sqrt(2*gama));
    [anew,~] = EProjSimplex_new(dnew,sqrt(2*gama));
    A(:,i) = (anew./sqrt(2*gama));
end
