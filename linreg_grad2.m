function [theta,J,rmse] = linreg_grad2(X,y,alpha,numIter,lambda,normalize)
%% adds the ’intercept terms’ to training data
if sum(X(:,1)==1)~=size(X,1)
    X = [ones(size(X,1),1) X];
end


if normalize
%% applies feature scaling (mean and standard deviation)
    [Xfeat,mu_x,sigma_x] = feat_scaling(X);
else
    Xfeat = X;
end


%% regression parameters
m = size(X,1); % number of training examples
theta = zeros(size(X,2),1); % initial ’guess’ for the solution
J = nan(numIter,2); % array for cost function values (for each iteration)
J(:,1) = (1:numIter)'; % store iteration steps

%% Gradient Descent algorithm (with regularization) for a fixed number of iterations
for i=1:numIter
    J(i,2) = 1/(2*m)*sum((Xfeat*theta-y).^2);
    grad = (1/m*(Xfeat*theta-y)'*Xfeat)';
    theta = theta - alpha*(grad+lambda/m*[0;theta(2:end)]);
end

if normalize
%% Returns the non-normalized parameter solution
    theta(1) = theta(1)-sum(mu_x.*theta(2:end)'./sigma_x);
    theta(2:end) = theta(2:end)./sigma_x';
end


%% compute the predicted RMSE (root mean square error)
rmse = sqrt(mean((X*theta-y).^2));
end

