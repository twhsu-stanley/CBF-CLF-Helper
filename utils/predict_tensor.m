function f = predict_tensor(x, u, feature_names, coefficients)
% Compute the model predciction using expressions for torch tensor operations
%   x: array or tensor (batch_size x n_states)
%   u: array or tensor (batch_size x n_controls)
%   feature_names: list (len = n_features)
%   coefficients: array (size = n_states x n_features)

n_features = length(feature_names);
n_states = size(coefficients, 1);
n_controls = size(u, 1);

for s = 1:n_states
    eval("x"+num2str(s-1)+"=x(s)");
end

for s = 1:n_controls
    eval("u"+num2str(s-1)+"=u(s)");
end

f = zeros(1, n_states); %dtype = torch.float64
for s = 1:n_states
    for i = 1:n_features
        f(s) = f(s) + eval(feature_names(i)) * coefficients(s,i);
    end
end


end