function cbf = defineCbf(obj, params, symbolic_state)
    x = symbolic_state;
    x0 = x(1); x1 = x(2); x2 = x(3);

    xo = params.xo;
    yo = params.yo;
    d = params.d;

    distance = (x0 - xo)^2 + (x1 - yo)^2 - d^2;

    feature_names = params.feature_names;
    coefficients = params.coefficients;
    idx_x = params.idx_x;

    f1 = 0;
    for i = idx_x
        f1 = f1 + eval(feature_names(i)) * coefficients(1,i);
    end

    f2 = 0;
    for i = idx_x
        f2 = f2 + eval(feature_names(i)) * coefficients(2,i);
    end

    derivDistance = 2 * (x0-xo) * f1 + 2 * (x1-yo) * f2;
    cbf = derivDistance + params.cbf_gamma0 * distance;
end