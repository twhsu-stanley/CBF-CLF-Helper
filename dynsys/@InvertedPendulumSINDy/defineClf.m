function clf = defineClf(obj, params, symbolic_state)
    x = symbolic_state;

    % Numerically compute the A matrix of the learned model
    feature_names = params.feature_names;
    coefficients = params.coefficients;
    idx_x = params.idx_x;
    idx_u = params.idx_u; 
    syms x0 x1
    
    f1 = 0;
    f2 = 0;
    for i = idx_x
        f1 = f1 + eval(feature_names(i)) * coefficients(1,i);
        f2 = f2 + eval(feature_names(i)) * coefficients(2,i);
    end

    u0 = 1.0;
    g1 = 0;
    g2 = 0;
    for i = idx_u
        g1 = g1 + eval(feature_names(i)) * coefficients(1,i);
        g2 = g2 + eval(feature_names(i)) * coefficients(2,i);
    end
    
    A = simplify(jacobian([f1;f2], x));
    x0 = 0; x1 = 0;
    A = eval(A); % evaluate A at the equilibrium point
    B = [g1; g2];

    % LQR
    %{
    Q = params.clf.Q;
    R = params.clf.R;
    [K_lqr, P_lqr, ~] = lqr(A, B, Q, R);
    obj.K_lqr = K_lqr;
    obj.P_lqr = P_lqr;
    clf = x' * P_lqr * x;
    %}

    % Linearized Dynamics with state feedback : u0 = params.Kp * x0 + params.Kd * x1
    A_cl = A + B * [params.Kp, params.Kd];
    Q = params.clf.rate * eye(size(A,1));
    P = lyap(A_cl', Q); % Cost Matrix for quadratic CLF. (V = e'*P*e)
    clf = x' * P * x;
    
    % Find c1: c1*||x||^2 <= V(x) = x'Px <= c2*||x||^2
    obj.c1 = min(eig(P));
    obj.c2 = max(eig(P));
    %obj.c3 = params.clf.rate * obj.c1;
end 
