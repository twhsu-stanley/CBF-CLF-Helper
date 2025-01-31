classdef InvertedPendulumSINDy < CtrlAffineSys
    properties
        K_lqr
    end
    
    methods
        function [x, f, g] = defineSystem(obj, params)
            syms x0 x1
            x = [x0; x1];
            
            feature_names = params.feature_names;
            coefficients = params.coefficients;
            idx_x = params.idx_x;
            idx_u = params.idx_u;

            if size(coefficients, 1) ~= 2
                error("State dimension does not match with the coefficient matrix");
            end

            f = [];
            for s = 1:2
                fs = 0;
                for i = idx_x
                    fs = fs + eval(feature_names(i)) * coefficients(s,i);
                end
                f = [f; fs];
            end

            g = [];
            u0 = 1.0;
            for s = 1:2
                gs = 0;
                for i = idx_u
                    gs = gs + eval(feature_names(i)) * coefficients(s,i);
                end
                g = [g; gs];
            end
        end
    end
end
