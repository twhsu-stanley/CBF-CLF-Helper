classdef DubinsCarSINDy < CtrlAffineSys
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
            %f = [params.v * cos(x3);
            %     params.v * sin(x3);
            %     0];

            g = [];
            u0 = 1.0;
            for s = 1:3
                gs = 0;
                for i = idx_u
                    gs = gs + eval(feature_names(i)) * coefficients(s,i);
                end
                g = [g; gs];
            end
            %g = [0; 0; 1];
        end

        u_ref = ctrlNominal(obj, x, x_d, Kp)
    end 
end