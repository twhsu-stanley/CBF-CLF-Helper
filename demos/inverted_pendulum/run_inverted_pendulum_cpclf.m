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
% NOTE: SINDy model with ps.PolynomialLibrary(degree = 2) worked well

%% Set up the learned and true models
% Model Parameters
params.l = 1;    % [m]        length of pendulum
params.m = 1;    % [kg]       mass of pendulum
params.g = 9.81; % [m/s^2]    acceleration of gravity
params.b = 0.01; % [s*Nm/rad] friction coefficient

%params.u_max = 50;
%params.u_min = -params.u_max;

params.clf.rate = 1;
params.weight.slack = 400;
params.weight.input = 6;

% Learned model
params.feature_names = feature_names;
params.coefficients = coefficients;
params.idx_x = idx_x;
params.idx_u = idx_u;
ip_learned = InvertedPendulumSINDy(params);
controller_clf = @ip_learned.ctrlClfQp;
controller_cpclf = @ip_learned.ctrlCpClfQp;
%cp_quantile = 0; % setting cp_quantile = 0 is equivalent to using the regular clf

% True model
ip_true = InvertedPendulum(params);
dyn_true = @ip_true.dynamics;
%controller_clf = @ip_true.ctrlClfQp;
odeSolver = @ode45; %113
odeFun = dyn_true;

%% Create a grid of states and sample initiall states from it
resolution = 100;
x_ = linspace(-pi/4, pi/4, resolution);
y_ = linspace(-5, 5, resolution);
state = zeros(resolution, resolution, 2);
state_norm_square = zeros(resolution, resolution);
V_ = zeros(resolution, resolution);
for i = 1:resolution
    for j = 1:resolution
        state_norm_square(i,j) = x_(i)^2 + y_(j)^2;
        V_(i,j) = ip_learned.clf([x_(i); y_(j)]);
    end
end
clf_level = min([V_(1,:), V_(end,:), V_(:,1)', V_(:,end)']);

x0 = [];
for i = 1:resolution
    for j = 1:resolution
        if V_(i,j) <= clf_level && V_(i,j) >= clf_level - 0.01
            x0 = [x0; [x_(i), y_(j)]];
        end
    end
end

% Sample around the level set ip_learned.clf == clf_level as x0 
N = 10; % number of paths
N = min(N, size(x0, 1));
x0 = x0(randperm(length(x0)), :); % random shuffling
x0 = x0(1:N,:);

% Find c1: c1*||x||^2 <= V(x) <= c2*||x||^2
ind_roa = find(V_ <= clf_level);
c1 = min(V_(ind_roa) ./ state_norm_square(ind_roa));
c2 = max(V_(ind_roa) ./ state_norm_square(ind_roa));

% Calculate the parameters of exponential stability (M and gamma)
V0 = clf_level;
M = V0/c1;

% Testing
%N = 1;
%x0 = [pi/2, 0];
%V0 = ip_learned.clf([pi/2; 0]);

%% Run simulation
dt = 0.001;
T = 3;
tt = 0:dt:T;

% Time history
x_hist = zeros(N, length(tt), ip_true.xdim);
x_norm_hist = zeros(N, length(tt));
u_hist = zeros(N, length(tt)-1);
V_hist = zeros(N, length(tt)-1);
p_hist = zeros(N, length(tt)-1);
p_cp_hist = zeros(N, length(tt)-1);
slack_hist = zeros(N, length(tt)-1);
model_err__hist = zeros(N, length(tt)-1);

for n = 1:N
    for k = 1:length(tt)-1
        if k == 1
            x_hist(n, 1, :) = x0(n,:)';
            x_norm_hist(n, 1) = norm(x0(n,:)');
        end

        % Wrap angle to pi
        % *** This is crucial ***
        x_hist(n, k, 1) = wrapToPi(x_hist(n, k, 1));

        t = tt(k);
        x = squeeze(x_hist(n, k, :));

        % Controller
        %[u, slack, V, feas, comp_time] = controller_clf(x);
        u_ref = 0; % -ip_learned.K_lqr * x; %
        [u, slack, V, feas]  = controller_cpclf(x, u_ref, cp_quantile);

        u_hist(n, k) = u;
        if isempty(slack)
            slack = 0;
        end
        slack_hist(n, k) = slack;
        V_hist(n, k) = V;

        p_hist(n, k) = ip_learned.dclf(x) * (ip_true.f(x) + ip_true.g(x) * u) + params.clf.rate * ip_learned.clf(x);
        p_cp_hist(n, k) = ip_learned.dclf(x) * (ip_learned.f(x) + ip_learned.g(x) * u) + params.clf.rate * ip_learned.clf(x)...
            + cp_quantile * norm(ip_learned.dclf(x), 2);
        model_err__hist(n, k) = norm(ip_true.f(x) + ip_true.g(x) * u - ip_learned.f(x) - ip_learned.g(x) * u, 2);

        % Run one time step propagation.
        %x_hist(n, k+1, :) = x + dyn_true(t, x, u) * dt;
        [ts_temp, xs_temp] = odeSolver(@(t, s) odeFun(t, s, u), [t t+dt], x);
        x_hist(n, k+1, :) = xs_temp(end, :);
        x_norm_hist(n, k+1) = norm(squeeze(x_hist(n, k+1, :)));

    end
end

%% Plots
figure;
title('Inverted Pendulum: CLF-QP States');
subplot(2, 1, 1);
plot(tt, squeeze(x_hist(:,:,1))); grid on
xlabel('Time (s)'); ylabel("theta (rad)"); 
%plot(tt, squeeze(180 * x_hist(:,:,1)/pi)); grid on
%xlabel('Time (s)'); ylabel("$\theta$ (deg)",'interpreter','latex'); 
subplot(2, 1, 2);
plot(tt, squeeze(x_hist(:,:,2))); grid on
xlabel('Time (s)'); ylabel("theta dot (rad/s)");
%plot(tt, squeeze(180 * x_hist(:,:,2)/pi)); grid on
%xlabel('Time (s)'); ylabel("$\dot{\theta}$ (deg/s)",'interpreter','latex'); 

figure;
plot(tt(1:end-1), u_hist); hold on
xlabel('Time (s)'); 
ylabel('ut');
grid on

figure;
plot(tt(1:end-1), slack_hist); hold on
xlabel('Time (s)'); 
ylabel('slack');
grid on

figure;
plot(tt(1:end-1), model_err__hist); hold on
xlabel('Time (s)'); 
ylabel('||model err||');
grid on

figure;
plot(tt, x_norm_hist); hold on
plot(tt(1:end-1), sqrt(M) * exp(-params.clf.rate/2 * tt(1:end-1)), 'r--');
%plot(tt(1:end-1), sqrt(c2/c1) * x_norm_hist(:,1) * exp(-params.clf.rate/2 * tt(1:end-1)), 'g--');
xlabel('Time (s)'); 
ylabel('State norm: ||x||');
grid on

figure;
plot(tt(1:end-1), V_hist); hold on
plot(tt(1:end-1), V0 * exp(-params.clf.rate * tt(1:end-1)), 'r--');
xlabel('Time (s)'); 
ylabel('CLF: V(x_t)');
grid on

figure;
plot(tt(1:end-1), p_hist, '-b'); hold on
%plot(tt(2:end-1), diff(V_hist,1,2)/dt, 'r--'); % checking
plot(tt(1:end-1), p_cp_hist, '-r'); hold on
xlabel('Time (s)'); 
ylabel('pCLF');
grid on