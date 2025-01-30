clc
clear
close all

dt = 0.01;
T = 5;
N = 50; % number of paths

%% Model Parameters
params.l = 1;    % [m]        length of pendulum
params.m = 1;    % [kg]       mass of pendulum
params.g = 9.81; % [m/s^2]    acceleration of gravity
params.b = 0.01; % [s*Nm/rad] friction coefficient

%params.u_max = 7;
%params.u_min = -params.u_max;

params.I = params.m*params.l^2/3; 

% Assumed feedback gains, used to construct a CLF.
params.Kp = 6;
params.Kd = 5;

params.clf.rate = 2.5;
params.weight.slack = 100000;

x0 = [pi/2; 0.0];

ip_sys = InvertedPendulum(params);

dyn_fg = @ip_sys.dynamics;
controller_clf = @ip_sys.ctrlClfQp;
controller_sclf = @ip_sys.ctrlStochasticClfQp;

tt = 0:dt:T;
% initialize traces.
xs = zeros(N, length(tt), ip_sys.xdim);
us = zeros(N, length(tt)-1);
Vs = zeros(N, length(tt)-1);

sigma = [0.1; 0.1];

for n = 1:N
    for k = 1:length(tt)-1
        if k == 1
            xs(n, 1, :) = x0';
        end

        t = tt(k);
        x = squeeze(xs(n, k, :));

        % Determine control input.
        % dV_hat: analytic Vdot based on model.
        %[u, slack, V, feas, comp_time] = controller_clf(x);
        [u, slack, V, feas, comp_time]  = controller_sclf(x, 0, sigma);

        us(n, k) = u;
        Vs(n, k) = V;

        % Run one time step propagation.
        xs(n, k+1, :) = x + dyn_fg(t, x, u) * dt + sqrt(dt) * sigma * randn;
    end
end

figure;
title('Inverted Pendulum: CLF-QP States');
subplot(2, 1, 1);
plot(tt, squeeze(180 * xs(:,:, 1)/pi)); grid on
xlabel('Time (s)'); ylabel("$\theta$ (deg)",'interpreter','latex'); 
subplot(2, 1, 2);
plot(tt, squeeze(180 * xs(:,:, 2)/pi)); grid on
xlabel('Time (s)'); ylabel("$\dot{\theta}$ (deg/s)",'interpreter','latex'); 

figure
plot(tt(1:end-1), Vs)
xlabel('Time (s)');
ylabel('SCLF: V(x_t)');
grid on