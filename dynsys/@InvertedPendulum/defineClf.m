function clf = defineClf(obj, params, symbolic_state)
    x = symbolic_state;
    I = params.m * params.l^2 / 3;
    c_bar = params.m*params.g*params.l/(2*I);
    b_bar = params.b/I;
    
    % Linearized Dynamics with state feedback : u0 = params.Kp * x0 + params.Kd * x1
    A = [0, 1;
         c_bar-params.Kp/I, -b_bar-params.Kd/I];
    Q = params.clf.rate * eye(size(A,1));
    P = lyap(A', Q); % Cost Matrix for quadratic CLF. (V = e'*P*e)
    clf = x' * P * x;
    
    % LQR
    %{
    A = [0, 1;
         c_bar, -b_bar];
    B = [0; -1/I];
    Q = params.clf.Q;
    R = params.clf.R;
    [K_lqr, P_lqr, ~] = lqr(A, B, Q, R);
    obj.K_lqr = K_lqr;
    obj.P_lqr = P_lqr;
    clf = x' * P_lqr * x;
    %}
end
