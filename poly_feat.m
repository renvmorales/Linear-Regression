% X = load('ex1data2.txt');
% X = X(:,1:2);  % x1 | x2 | ... | xn | y
% 
% n = size(X,2); % number of features
% d = 2; % polynomial degree

function Xpoly = poly_feat(X,d)
if sum(X(:,1)==1)==size(X,1)
    X = X(:,2:end);
end
if d==1  % preserves the same training data matrix for d=1
    Xpoly = [ones(size(X,1),1) X];
    return
end

    
n = size(X,2); % number of features

Xblck = nan*ones(size(X,1),1e5,'double');
ct_blck = 0; % count column blocks
% Xblck = [];


for i=2:d  % spanning on different combination-tuples (cross mult. terms)
    col_comb = combnk(1:n,i); % all possible combinations    
%% exits loop if number of grouping elem. exceeds number of feat.
    if isempty(col_comb) 
        break
    end    
    pwr = combnk(repmat(1:d,1,i),i);
    pwr = unique(pwr(sum(pwr,2)<=d,:),'rows'); % powers to apply 
    for j=1:size(pwr,1)
        for k=1:size(col_comb,1)
            ct_blck = ct_blck+1;
            Xblck(:,ct_blck) = prod(bsxfun(@power,X(:,col_comb(k,:)),pwr(j,:)),2);
        end
    end    
end

Xblck = Xblck(:,all(~isnan(Xblck))); % discards extra space

% includes 'single' power terms
Xblck = [Xblck bsxfun(@power,repmat(X,1,(d-1)),sort(repmat(2:d,1,n)))];


%% adding the ’intercept term’ into the training data matrix
Xpoly = [ones(size(X,1),1) X Xblck];
end

