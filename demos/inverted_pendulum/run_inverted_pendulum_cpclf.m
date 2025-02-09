clc
clear
close all

%% Loading the SYNDy model from Python
pickle = py.importlib.import_module('pickle');
%fh = py.open('..\..\..\pysindy\control_affine_models\saved_models\model_inverted_pendulum_sindy', 'rb');
fh = py.open('..\..\sindy_models\model_inverted_pendulum_sindy', 'rb');
P = pickle.load(fh); % pickle file loaded to Python variable
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

%% Use conformal prediction or not
use_cp = 1; % 1 or 0: whether to use conformal prediction
cp_quantile_a = cp_quantile; % for robustness analysis
cp_quantile = cp_quantile * use_cp; % setting cp_quantile = 0 is equivalent to using the regular clf

%% Set up the learned and true models
% Model Parameters
params.l = 1;    % [m]        length of pendulum
params.m = 1;    % [kg]       mass of pendulum
params.g = 9.81; % [m/s^2]    acceleration of gravity
params.b = 0.01; % [s*Nm/rad] friction coefficient

% CLF
%params.clf.Q = diag([1, 1e-2]);
%params.clf.R = 1;
params.clf.rate = 0.5;
params.Kp = 8;
params.Kd = 5;

% QP solver
%params.u_max = 50;
%params.u_min = -params.u_max;
if use_cp
    params.weight.slack = 500;
    %params.weight.input = 0.1;
else
    params.weight.slack = 500;
    %params.weight.input = 0.01;
end

% Learned model
params.feature_names = feature_names;
params.coefficients = coefficients;
params.idx_x = idx_x;
params.idx_u = idx_u;
ip_learned = InvertedPendulumSINDy(params);
controller_clf = @ip_learned.ctrlClfQp;
controller_cpclf = @ip_learned.ctrlCpClfQp;

% True model
ip_true = InvertedPendulum(params);
dyn_true = @ip_true.dynamics;
%controller_clf = @ip_true.ctrlClfQp;
odeSolver = @ode45;
odeFun = dyn_true;

%% Create a grid of states and sample initiall states from it
resolution = 200;
x_ = linspace(-pi/3, pi/3, resolution);
y_ = linspace(-pi*2, pi*2, resolution);
state = zeros(resolution, resolution, 2);
state_norm_square = zeros(resolution, resolution);
V_ = zeros(resolution, resolution);
gradV_ = zeros(resolution, resolution, 2);
for i = 1:resolution
    for j = 1:resolution
        state_norm_square(i,j) = x_(i)^2 + y_(j)^2;
        V_(i,j) = ip_learned.clf([x_(i); y_(j)]);
        gradV_(i,j,:) = ip_learned.dclf([x_(i); y_(j)]);
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

figure;
[X, Y] = meshgrid(x_, y_);
contourf(X, Y, V_', "ShowText", true); hold on
%colormap(summer)
colorbar
marker_size = 10;
scatter(x0(:,1), x0(:,2), marker_size, "filled", "MarkerFaceColor", [1, 0, 0]);
xlabel("theta (rad)");
ylabel("theta dot (rad/s)");

% Sample around the level set ip_learned.clf == clf_level as x0 
N = 20; % number of paths
N = min(N, size(x0, 1));
x0 = x0(randperm(length(x0)), :); % random shuffling
x0 = x0(1:N,:);

% Find c1: c1*||x||^2 <= V(x) <= c2*||x||^2
ind_roa = find(V_ <= clf_level);
c1 = min(V_(ind_roa) ./ state_norm_square(ind_roa));
c2 = max(V_(ind_roa) ./ state_norm_square(ind_roa));

% Estimate Lipchitz constant
Lv = max(vecnorm(gradV_(ind_roa), 2, 3),[],"all"); 

% Calculate the parameters of exponential stability (M and gamma)
V0 = clf_level;
M = V0/c1;

% Testing
%N = 1;
%x0 = [pi/2, 0];
%V0 = ip_learned.clf([pi/2; 0]);

%% Run simulation
dt = 0.001;
T = 5;
tt = 0:dt:T;

% Time history
x_hist = zeros(N, length(tt), ip_true.xdim);
x_norm_hist = zeros(N, length(tt));
u_hist = zeros(N, length(tt)-1);
V_hist = zeros(N, length(tt)-1);
p_hist = zeros(N, length(tt)-1);
p_hat_hist = zeros(N, length(tt)-1);
p_cp_hist = zeros(N, length(tt)-1);
slack_hist = zeros(N, length(tt)-1);
p_err_hist = zeros(N, length(tt)-1);
cp_bound_hist = zeros(N, length(tt)-1);

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
        %V_hist(n, k) = x' * ip_learned.P_lqr * x;

        p_hist(n, k) = ip_learned.dclf(x) * (ip_true.f(x) + ip_true.g(x) * u) + params.clf.rate * ip_learned.clf(x);
        p_hat_hist(n, k) = ip_learned.dclf(x) * (ip_learned.f(x) + ip_learned.g(x) * u) + params.clf.rate * ip_learned.clf(x);
        p_cp_hist(n, k) = ip_learned.dclf(x) * (ip_learned.f(x) + ip_learned.g(x) * u) + params.clf.rate * ip_learned.clf(x)...
                          + cp_quantile * norm(ip_learned.dclf(x), 2);
        p_err_hist(n, k) = ip_learned.dclf(x) * (ip_true.f(x) + ip_true.g(x) * u - ip_learned.f(x) - ip_learned.g(x) * u);
        cp_bound_hist(n, k) = cp_quantile * norm(ip_learned.dclf(x), 2);

        % Run one time step propagation.
        %x_hist(n, k+1, :) = x + dyn_true(t, x, u) * dt;
        [ts_temp, xs_temp] = odeSolver(@(t, s) odeFun(t, s, u), [t t+dt], x);
        x_hist(n, k+1, :) = xs_temp(end, :);
        x_norm_hist(n, k+1) = norm(squeeze(x_hist(n, k+1, :)));
        %x_norm_hist(n, k) = norm(x);
    end
end

%% Plots
figure;
subplot(2, 1, 1);
plot(tt, squeeze(x_hist(:,:,1))); grid on
xlabel('Time (s)'); ylabel("theta (rad)"); 
%plot(tt, squeeze(180 * x_hist(:,:,1)/pi)); grid on
%xlabel('Time (s)'); ylabel("$\theta$ (deg)","interpreter","latex"); 
subplot(2, 1, 2);
plot(tt, squeeze(x_hist(:,:,2))); grid on
xlabel('Time (s)'); ylabel("theta dot (rad/s)");
%plot(tt, squeeze(180 * x_hist(:,:,2)/pi)); grid on
%xlabel('Time (s)'); ylabel("$\dot{\theta}$ (deg/s)","interpreter","latex');
if use_cp
    exportgraphics(gcf, "plots/cpclf_inverted_pendulum_states.pdf","Resolution",500);
else
    exportgraphics(gcf, "plots/clf_inverted_pendulum_states.pdf","Resolution",500);
end

figure;
[X, Y] = meshgrid(x_, y_);
contourf(X, Y, V_',"ShowText",true); hold on
for n = 1:N
    plot(squeeze(x_hist(n,:,1)), squeeze(x_hist(n,:,2)), "LineWidth", 1.5); hold on
end
xlabel("theta (rad)");
ylabel("theta dot (rad/s)");
if use_cp
    exportgraphics(gcf, "plots/cpclf_inverted_pendulum_2d.pdf","Resolution",500);
else
    exportgraphics(gcf, "plots/clf_inverted_pendulum_2d.pdf","Resolution",500);
end

figure;
plot(tt(1:end-1), u_hist); hold on
xlabel('Time (s)'); 
ylabel('Control: ut');
grid on
if use_cp
    exportgraphics(gcf, "plots/cpclf_inverted_pendulum_control.pdf","Resolution",500);
else
    exportgraphics(gcf, "plots/clf_inverted_pendulum_control.pdf","Resolution",500);
end

figure;
plot(tt(1:end-1), slack_hist); hold on
xlabel('Time (s)'); 
ylabel('QP slack');
grid on
if use_cp
    exportgraphics(gcf, "plots/cpclf_inverted_pendulum_qpslack.pdf","Resolution",500);
else
    exportgraphics(gcf, "plots/clf_inverted_pendulum_qpslack.pdf","Resolution",500);
end

if use_cp
    figure;
    plot(tt(1:end-1), p_err_hist, 'b-'); hold on
    plot(tt(1:end-1), cp_bound_hist, 'r-'); hold on
    xlabel('Time (s)'); 
    %ylabel('');
    grid on
end

figure;
plot(tt, x_norm_hist); hold on
%plot(tt(1:end-1), sqrt(M) * exp(-params.clf.rate/2 * tt(1:end-1)), 'r--');
%plot(tt(1:end-1), sqrt(c2/c1) * x_norm_hist(:,1) * exp(-params.clf.rate/2 * tt(1:end-1)), 'g--');
ylim([0, inf]);
xlabel('Time (s)'); 
ylabel('State norm: ||x||');
grid on
if use_cp
    exportgraphics(gcf, "plots/cpclf_inverted_pendulum_state_norm.pdf","Resolution",500);
else
    exportgraphics(gcf, "plots/clf_inverted_pendulum_state_norm.pdf","Resolution",500);
end

figure;
plot(tt(1:end-1), V_hist); hold on
plot(tt(1:end-1), V0 * exp(-params.clf.rate * tt(1:end-1)), 'r--', 'LineWidth',1.5); hold on
xlabel('Time (s)');
grid on
if use_cp
    ylabel('CP-CLF: V(x_t)');
    exportgraphics(gcf, "plots/cpclf_inverted_pendulum_cpclf.pdf","Resolution",800);
else
    %plot(tt(1:end-1), V0 * exp(-params.clf.rate * tt(1:end-1)) + Lv*cp_quantile_a/params.clf.rate * (1-exp(-params.clf.rate * tt(1:end-1))), 'b--');
    ylabel('CLF: V(x_t)');
    exportgraphics(gcf,"plots/clf_inverted_pendulum_clf.pdf","Resolution",800)
end

figure;
subplot(3,1,1);
plot(tt(1:end-1), p_hist); hold on
%plot(tt(2:end-1), diff(V_hist,1,2)/dt, 'r--'); % checking
grid on
xlabel('Time (s)'); 
ylabel('pCLF');
subplot(3,1,2);
plot(tt(1:end-1), p_hat_hist); hold on
grid on
xlabel('Time (s)'); 
ylabel('pCLF hat');
subplot(3,1,3);
plot(tt(1:end-1), p_cp_hist); hold on
grid on
xlabel('Time (s)');
ylabel('pCLF-CP');
if use_cp
    exportgraphics(gcf, "plots/cpclf_inverted_pendulum_pclf.pdf","Resolution",500);
else
    exportgraphics(gcf, "plots/clf_inverted_pendulum_pclf.pdf","Resolution",500);
end
