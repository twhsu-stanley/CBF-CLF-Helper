clc
clear
close all

%% Loading the SYNDy model from Python
pickle = py.importlib.import_module('pickle');
fh = py.open('..\..\..\pysindy\control_affine_models\saved_models\model_inverted_pendulum_sindy', 'rb');
%fh = py.open('..\..\sindy_models\model_inverted_pendulum_sindy', 'rb');
P = pickle.load(fh);    % pickle file loaded to Python variable
fh.close();

feature_names = string(P{'feature_names'}); % TODO: rename
coefficients = double(py.array.array('d',py.numpy.nditer(P{"coefficients"}))); % TODO: reshape
n_dim = double(P{"coefficients"}.shape(1));
n_features = length(feature_names);
coefficients = reshape(coefficients, [n_features, n_dim])';

cp_quantile = P{'model_error'}{'quantile'};
fprintf("cp_quantile = %5.3f \n", cp_quantile);

idx_x = []; % Indices for f(x)
idx_u = []; % Indices for g(x)*u
for i = 1: length(feature_names)
    if contains(feature_names(i), 'u0')
        idx_u = [idx_u, i];
    else
        idx_x = [idx_x, i];
    end

    feature_names = replace(feature_names, " ", "*");
end
% NOTE: SINDy 

%% Implementation of the CP-CLF
dt = 0.001;
T = 5;
tt = 0:dt:T;

N = 1; % number of paths
x0 = [pi/5; 0.0];

% Model Parameters
params.l = 1;    % [m]        length of pendulum
params.m = 1;    % [kg]       mass of pendulum
params.g = 9.81; % [m/s^2]    acceleration of gravity
params.b = 0.01; % [s*Nm/rad] friction coefficient

%params.u_max = 10;
%params.u_min = -params.u_max;

params.I = params.m*params.l^2/3; 

% Assumed feedback gains, used to construct a CLF.
params.Kp = 6;
params.Kd = 5;

params.clf.rate = 1.0;
params.weight.slack = 100;

% Learned model
params.feature_names = feature_names;
params.coefficients = coefficients;
params.idx_x = idx_x;
params.idx_u = idx_u;
ip_learned = InvertedPendulumSINDy(params);
controller_clf = @ip_learned.ctrlClfQp;
controller_cpclf = @ip_learned.ctrlCpClfQp;
%cp_quantile = 0;

% True model
ip_true = InvertedPendulum(params);
dyn_true = @ip_true.dynamics;
%controller_clf = @ip_true.ctrlClfQp;

% Time history
xs = zeros(N, length(tt), ip_true.xdim);
us = zeros(N, length(tt)-1);
Vs = zeros(N, length(tt)-1);

for n = 1:N
    for k = 1:length(tt)-1
        if k == 1
            xs(n, 1, :) = x0';
        end

        % Wrap angle to pi
        % *** This is crucial ***
        xs(n, k, 1) = wrapToPi(xs(n, k, 1));

        t = tt(k);
        x = squeeze(xs(n, k, :));

        % Determine control input.
        % dV_hat: analytic Vdot based on model.
        %[u, slack, V, feas, comp_time] = controller_clf(x);
        [u, slack, V, feas, comp_time]  = controller_cpclf(x, 0, cp_quantile);

        us(n, k) = u;
        Vs(n, k) = V;

        % Run one time step propagation.
        xs(n, k+1, :) = x + dyn_true(t, x, u) * dt;
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
plot(tt(1:end-1), Vs); hold on
plot(tt(1:end-1), Vs(1) * exp(-params.clf.rate * tt(1:end-1)), 'r--');
xlabel('Time (s)'); 
ylabel('CP-CLF: V(x_t)');
grid on