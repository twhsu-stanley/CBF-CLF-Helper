function u = ctrlNominal(obj, v, v_d, Kp)
    % Nominal controller that tracks desired speed
    u = Kp * (v_d - v);
end