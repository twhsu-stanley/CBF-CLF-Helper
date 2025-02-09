classdef ACCSINDy < CtrlAffineSys    
    methods        
        function [x, f, g] = defineSystem(obj, params)
            syms x0 x1 x2
            x = [x0; x1; x2];
            
            feature_names = params.feature_names;
            coefficients = params.coefficients;
            idx_x = params.idx_x;
            idx_u = params.idx_u;

            if size(coefficients, 1) ~= 3
                error("State dimension does not match with the coefficient matrix");
            end

            f = [];
            for s = 1:3
                fs = 0;
                for i = idx_x
                    fs = fs + eval(feature_names(i)) * coefficients(s,i);
                end
                f = [f; fs];
            end

            g = [];
            u0 = 1.0;
            for s = 1:3
                gs = 0;
                for i = idx_u
                    gs = gs + eval(feature_names(i)) * coefficients(s,i);
                end
                g = [g; gs];
            end
        end

        function Fr = getFr(obj, x)
            v = x(2);
            Fr = obj.params.f0 + obj.params.f1 * v + obj.params.f2 * v^2;
        end

        u_ref = ctrlNominal(obj, v, v_d, Kp)
    end
end
