function clf = defineClf(~, params, symbolic_state)
    p_x = symbolic_state(1);
    p_y = symbolic_state(2);
    theta = symbolic_state(3);
    clf = (cos(theta).*(p_y-params.yd)-sin(theta).*(p_x-params.xd)).^2;
    %clf = [p_x-params.xd , p_y-params.yd, theta] * 5*eye(3) * [p_x-params.xd ; p_y-params.yd; theta];
end