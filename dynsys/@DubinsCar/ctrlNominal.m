function u = ctrlNominal(obj, x, x_d, Kp)
    
    %if isempty(obj.cbf)
    %    error('CBF is not defined so ctrlCbfQp cannot be used. Create a class function [defineCbf] and set up cbf with symbolic expression.');
    %end
        
    theta_d = atan2(x_d(2) - x(2), x_d(1) - x(1));

    theta_err = theta_d - x(3);
    theta_err = wrapToPi(theta_err);

    u = Kp * theta_err;
    
end