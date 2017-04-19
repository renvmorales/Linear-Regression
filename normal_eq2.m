function [theta,rmse] = normal_eq2(X,y,lambda,normalize)
%% adds the ’intercept’ term to training data
if sum(X(:,1)==1)~=size(X,1)
    X = [ones(size(X,1),1) X];
end

if normalize
%% applies feature scaling (mean and standard deviation)
    [Xfeat,mu_x,sigma_x] = feat_scaling(X);
else
    Xfeat = X;
end


%% Regression paramaters
n = size(X,2) - 1; % number of features
A = Xfeat'*Xfeat+lambda*diag([0 ones(1,n)]); % Matrix to be inverted

%% computes the theoretical solution using the pseudoinverse
theta = A\(Xfeat'*y);
% theta = pinv(A)*(Xfeat'*y);


if normalize
%% Returns the non-normalized parameter solution
    theta(1) = theta(1)-sum(mu_x.*theta(2:end)'./sigma_x);
    theta(2:end) = theta(2:end)./sigma_x';
end

%% computes the the predicted RMSE (root mean square error)
rmse = sqrt(mean((X*theta-y).^2));
end