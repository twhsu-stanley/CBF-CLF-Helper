function cbf = defineCbf(obj, params, symbolic_state)
    v = symbolic_state(2);
    z = symbolic_state(3);

    T = params.T;
    
    cbf = z - T * v;
end