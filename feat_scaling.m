function [Xfeat,mu_x,sigma_x] = feat_scaling(X)
%% discard 'intercept terms'
extra_col=false; % 'true' when intercept terms exists
if sum(X(:,1)==1)==size(X,1)
    X = X(:,2:end); % discard 'intercept terms'
    extra_col = true; 
end


%% applies feature scaling
mu_x = mean(X,1); % computes the mean of all features
sigma_x = std(X,0,1); % computes the std of all features
X = bsxfun(@minus,X,mu_x); % mean of data is reduced to zero 
X = bsxfun(@rdivide,X,sigma_x); % data normalization 


%% output data will be the same format as the input
if extra_col % recovers the 'intercept terms'
    Xfeat = [ones(size(X,1),1) X];
else  
    Xfeat = X;
end
end