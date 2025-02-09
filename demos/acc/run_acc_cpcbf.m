clc
clear
close all

%% Loading the SYNDy model from Python
pickle = py.importlib.import_module('pickle');
%fh = py.open('..\..\..\pysindy\control_affine_models\saved_models\model_acc_sindy', 'rb');
fh = py.open('..\..\sindy_models\model_acc_sindy', 'rb');
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

%% Use conformal prediction or not
use_cp = 1; % 1 or 0: whether to use conformal prediction
cp_quantile = cp_quantile * use_cp; % setting cp_quantile = 0 is equivalent to using the regular cbf

%% Implementation of the CP-CBF
dt = 0.02;
sim_T = 8;
tt = 0:dt:sim_T;

% System parameters
params.v0 = 15;
params.vd = 20;
params.m  = 2000;
params.g = 9.81; % not used; for true model only
params.f0 = 0.5;
params.f1 = 5.0;
params.f2 = 1.0;
params.ca = 0.3; % not used; for true model only
params.cd = 0.3; % not used; for true model only
params.T = 1.0;

% CBF
params.cbf.rate = 2;

% QP solver
%params.u_max = params.ca * params.m * params.g;
%params.u_min  = -params.cd * params.m * params.g;
params.weight.input = 2/params.m^2;

% Learned model
params.feature_names = feature_names;
params.coefficients = coefficients;
params.idx_x = idx_x;
params.idx_u = idx_u;
acc_learned = ACCSINDy(params);
controller_nominal = @acc_learned.ctrlNominal; % proportional navigation
Kp = 100; % gain for the proportional navigation
controller_cpcbf = @acc_learned.ctrlCpCbfQp;
controller_cbf = @acc_learned.ctrlCbfQp;

% True model
acc_true = ACC(params);
dyn_true = @acc_true.dynamics;

% Sample initial states within the safe set
N = 20; % number of paths
rand_temp = rand(1,N);
x0 = [rand_temp * 0; 
      rand_temp * 10 + params.vd;
      params.T*(rand_temp * 10 + params.vd) + rand(1,N) * 2 + 0.5]; % initial states

% Time history
x_hist = zeros(N, length(tt), 3);
u_hist = zeros(N, length(tt)-1, 1);
h_shist = zeros(N, length(tt)-1, 1);

for n = 1:N
    for k = 1:length(tt)-1
        if k == 1 
            x_hist(n, 1, :) = x0(:,n)';
        end

        t = tt(k);
        x = squeeze(x_hist(n, k, :));

        % Determine control input
        u_ref = controller_nominal(x(2), params.vd, Kp);
        %[u, h, feas, comp_time] = controller_cbf(x, u_ref);
        [u, h, feas, comp_time] = controller_cpcbf(x, u_ref, cp_quantile);
        if feas == 0
            error("controller_cpcbf infeasible");
        end
        u_hist(n, k, :) = u';
        h_shist(n, k) = h;

        % Run one time step propagation.
        x_hist(n, k+1, :) = x + dyn_true(t, x, u) * dt;
    end
end

%% Plots
figure;
subplot(2,1,1);
plot(tt, squeeze(x_hist(:,:,2))); hold on
yline(params.vd, 'k--', 'LineWidth',1); hold on
yline(params.v0, 'b--', 'LineWidth',1);
ylabel("v (m/s)");
set(gca,'FontSize',14);
grid on;
subplot(2,1,2);
plot(tt, squeeze(x_hist(:,:,3)));
ylabel("z (m)");
set(gca, 'FontSize', 14);
grid on;
if use_cp
    exportgraphics(gcf, "plots/cpcbf_acc_states.pdf","Resolution",500);
else
    exportgraphics(gcf, "plots/cbf_acc_states.pdf","Resolution",500);
end

figure;
plot(tt(1:end-1), u_hist); hold on;
%plot(tt(1:end-1), params.u_max*ones(size(tt, 1)-1, 1), 'k--');
%plot(tt(1:end-1), params.u_min*ones(size(tt, 1)-1, 1), 'k--');
ylabel("Control Input: u (N)");
set(gca, 'FontSize', 14);
grid on;

figure;
for n = 1:N
    h = plot(tt(1:end-1), squeeze(h_shist(n,:,:))); hold on
    c = get(h, 'Color');
    set(h, 'Color', [c 0.9]);
end
yline(0, 'r-', 'LineWidth',2);
ylabel("CBF: h(x_t)");
set(gca, 'FontSize', 14);
grid on;
if use_cp
    exportgraphics(gcf, "plots/cpcbf_acc_cbf.pdf","Resolution",500);
else
    exportgraphics(gcf, "plots/cbf_acc_cbf.pdf","Resolution",500);
end