clc
clear
close all

%% Loading the SYNDy model from Python
pickle = py.importlib.import_module('pickle');
%fh = py.open('..\..\..\pysindy\control_affine_models\saved_models\model_dubins_car_sindy', 'rb');
fh = py.open('..\..\sindy_models\model_dubins_car_sindy', 'rb');
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
dt = 0.01;
T = 10;
tt = 0:dt:T;

% Desired target state
x_d = [2; 4; 0];
params.xd = x_d(1);
params.yd = x_d(2);

% System parameter
params.v = 1.0; % velocity

% Obstacle position
params.xo = 5;
params.yo = 4;
params.d = 2; % radius

% CBF
params.cbf.rate = 1;
params.cbf_gamma0 = 15;

% Sample initial states outside the obstacle
N = 100; % number of paths
x0 = [rand(1,N) * 8 + 2; rand(1,N) * 6 + 1; rand(1,N) * 2*pi - pi]; % initial state
x0 = x0(:, (x0(1,:) - params.xo).^2 + (x0(2,:) - params.yo).^2 > (params.d * 1.05)^2);
N = size(x0, 2);

% QP solver
%params.u_max = 50;
%params.u_min = -params.u_max;
params.weight.input = 10000; % default is 1

% Learned model
params.feature_names = feature_names;
params.coefficients = coefficients;
params.idx_x = idx_x;
params.idx_u = idx_u;
dubins_learned = DubinsCarSINDy(params);
controller_nominal = @dubins_learned.ctrlNominal; % proportional navigation
Kp = 10; % gain for the proportional navigation
controller_cpcbf = @dubins_learned.ctrlCpCbfQp;
controller_cbf = @dubins_learned.ctrlCbfQp;

% True model
dubins_true = DubinsCar(params);
dyn_true = @dubins_true.dynamics;
%controller_cbf = @dubins_true.ctrlCbfQp;
%controller_nominal = @dubins_true.ctrlNominal;

% Time history
x_hist = cell(N, 1);
u_hist = cell(N, 1);
h_hist = cell(N, 1);
p_hist = cell(N, 1);
p_cp_hist = cell(N, 1);

for n = 1:N
    x_s = zeros(length(tt), 3);
    u_s = zeros(length(tt)-1, 1);
    h_s = zeros(length(tt)-1, 1);
    p_s = zeros(length(tt)-1, 1);
    p_cp_s = zeros(length(tt)-1, 1);

    for k = 1:length(tt)-1
        if k == 1 
            x_s(1, :) = x0(:,n)';
        end

        % Wrap angle to pi
        % *** This is crucial ***
        x_s(k, 3) = wrapToPi(x_s(k, 3));

        t = tt(k);
        x = x_s(k, :)';

        if (x(1) - x_d(1)) ^2 + (x(2) - x_d(2)) ^2 < 0.01
            x_s = x_s(1:k-1, :);
            u_s = u_s(1:k-1, :);
            h_s = h_s(1:k-1, :);
            p_s = p_s(1:k-1, :);
            p_cp_s = p_cp_s(1:k-1, :);
            break
        end

        % Determine control input
        u_ref = controller_nominal(x, x_d, Kp);
        %[u, h, feas, comp_time] = controller_cbf(x, u_ref);
        [u, h, feas, comp_time] = controller_cpcbf(x, u_ref, cp_quantile);
        if feas == 0
            continue
        end

        u_s(k) = u;
        h_s(k) = h;

        p_s(k) = dubins_learned.dcbf(x) * (dubins_true.f(x) + dubins_true.g(x) * u) + params.cbf.rate * dubins_learned.cbf(x);
        p_cp_s(k) = dubins_learned.dcbf(x) * (dubins_learned.f(x) + dubins_learned.g(x) * u) ...
                          + params.cbf.rate * dubins_learned.cbf(x)...
                          - cp_quantile * norm(dubins_learned.dcbf(x), 2);

        % Run one time step propagation.
        x_s(k+1, :) = x + dyn_true(t, x, u) * dt;

    end
    x_hist{n} = x_s;
    u_hist{n} = u_s;
    h_hist{n} = h_s;
    p_hist{n} = p_s;
    p_cp_hist{n} = p_cp_s;
end


%% Plotting
p_o = [params.xo; params.yo];
r_o = params.d;

figure;
subplot(3,1,1);
for n = 1:N
    plot((0:size(x_hist{n},1)-1) * dt, x_hist{n}(:,1)); hold on
end
xlabel('Time (s)');
ylabel('x (m)');
grid on
subplot(3,1,2);
for n = 1:N
    plot((0:size(x_hist{n},1)-1) * dt, x_hist{n}(:,2)); hold on
end
xlabel('Time (s)');
ylabel('y (m)');
grid on
subplot(3,1,3);
for n = 1:N
    plot((0:size(x_hist{n},1)-1) * dt, x_hist{n}(:,3)); hold on
end
xlabel('Time (s)');
ylabel('theta (rad)');
grid on
if use_cp
    saveas(gcf, "plots/cpcbf_dubins_car_states.png");
else
    saveas(gcf, "plots/cbf_dubins_car_states.png");
end

figure;
for n = 1:N
    plot(x_hist{n}(:,1), x_hist{n}(:,2)); hold on
end
draw_circle(p_o, r_o); hold on
plot(x_d(1), x_d(2), 'r.', MarkerSize = 20); hold on
grid on
xlabel('x (m)');
ylabel('y (m)');
if use_cp
    saveas(gcf, "plots/cpcbf_dubins_car_2d.png");
else
    saveas(gcf, "plots/cbf_dubins_car_2d.png");
end

figure;
for n = 1:N
    plot((0:size(h_hist{n},1)-1) * dt, h_hist{n}(:,1)); hold on
end
xlabel('Time (s)');
grid on
if use_cp
    ylabel('CP-CBF: h(x_t)');
    saveas(gcf, "plots/cpcbf_dubins_car_cpcbf.png");
else
    ylabel('CBF: h(x_t)');
    saveas(gcf, "plots/cbf_dubins_car_cbf.png");
end

figure;
for n = 1:N
    plot((0:size(u_hist{n},1)-1) * dt, u_hist{n}(:,1)); hold on
end
xlabel('Time (s)');
ylabel('Control: u_t');
grid on
if use_cp
    saveas(gcf, "plots/cpcbf_dubins_car_control.png");
else
    saveas(gcf, "plots/cbf_dubins_car_control.png");
end

figure;
subplot(2,1,1);
for n = 1:N
    plot((0:size(p_hist{n},1)-1) * dt, p_hist{n}(:,1)); hold on
end
grid on
xlabel('Time (s)'); 
ylabel('pCBF');
subplot(2,1,2);
for n = 1:N
    plot((0:size(p_cp_hist{n},1)-1) * dt, p_cp_hist{n}(:,1)); hold on
end
grid on
xlabel('Time (s)');
ylabel('pCBF-CP');
if use_cp
    saveas(gcf, "plots/cpcbf_dubins_car_pcbf.png");
else
    saveas(gcf, "plots/cbf_dubins_car_pcbf.png");
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = draw_circle(center,r)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + center(1);
yunit = r * sin(th) + center(2);
plot(xunit, yunit, 'r-', 'LineWidth', 1.5); hold on
h = fill(xunit, yunit, 'r');
set(h, 'FaceAlpha', 0.4);
axis equal;
hold off

end
