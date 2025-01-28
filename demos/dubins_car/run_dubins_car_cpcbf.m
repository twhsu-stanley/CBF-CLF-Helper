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

%% Implementation of the CP-CBF
dt = 0.01;
T = 10;
tt = 0:dt:T;

N = 100; % number of paths
x0 = [rand(1,N) * 8 + 2; rand(1,N) * 6 + 1; rand(1,N) * 2*pi - pi]; % initial state
x_d = [2; 4; 0]; % desired state

params.v = 1.0; % velocity

%obj.params.u_max = pi/2;
%obj.params.u_min = -pi/2;

% Obstacle position
params.xo = 5;
params.yo = 4;

% Obstacle radius
params.d = 2;
params.cbf_gamma0 = 15;

% Exclude samples within the obstacle
x0 = x0(:, (x0(1,:) - params.xo).^2 + (x0(2,:) - params.yo).^2 > (params.d * 1.05)^2);
N = size(x0, 2);

% Desired target point
params.xd = x_d(1);
params.yd = x_d(2);

%params.clf.rate = 1;
params.weight.slack = 50; % for clf, not used for cbf

params.cbf.rate = 1;

% Learned model
params.feature_names = feature_names;
params.coefficients = coefficients;
params.idx_x = idx_x;
params.idx_u = idx_u;
%cp_quantile = 0; % setting cp_quantile = 0 is equivalent to regular cbf
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

for n = 1:N
    xs = zeros(length(tt), 3);
    us = zeros(length(tt)-1, 1);
    hs = zeros(length(tt)-1, 1);

    for k = 1:length(tt)-1
        if k == 1 
            xs(1, :) = x0(:,n)';
        end

        % Wrap angle to pi
        % *** This is crucial ***
        xs(k, 3) = wrapToPi(xs(k, 3));

        t = tt(k);
        x = xs(k, :)';

        if (x(1) - x_d(1)) ^2 + (x(2) - x_d(2)) ^2 < 0.01
            xs = xs(1:k-1, :);
            us = us(1:k-1, :);
            hs = hs(1:k-1, :);
            break
        end

        % Determine control input
        u_ref = controller_nominal(x, x_d, Kp);
        %[u, h, feas, comp_time] = controller_cbf(x, u_ref);
        [u, h, feas, comp_time] = controller_cpcbf(x, u_ref, cp_quantile);
        if feas == 0
            continue
        end

        us(k) = u;
        hs(k) = h;

        % Run one time step propagation.
        xs(k+1, :) = x + dyn_true(t, x, u) * dt;

    end
    x_hist{n} = xs;
    u_hist{n} = us;
    h_hist{n} = hs;
end


%% Plotting
p_o = [params.xo; params.yo];
r_o = params.d;

figure
subplot(3,1,1)
for n = 1:N
    plot((0:size(x_hist{n},1)-1) * dt, x_hist{n}(:,1)); hold on
end
xlabel('Time (s)');
ylabel('p_x (m)')
grid on

subplot(3,1,2)
for n = 1:N
    plot((0:size(x_hist{n},1)-1) * dt, x_hist{n}(:,2)); hold on
end
xlabel('Time (s)');
ylabel('p_y (m)')
grid on

subplot(3,1,3)
for n = 1:N
    plot((0:size(x_hist{n},1)-1) * dt, x_hist{n}(:,3)); hold on
end
xlabel('Time (s)');
ylabel('theta (rad)')
grid on

figure
for n = 1:N
    plot(x_hist{n}(:,1), x_hist{n}(:,2)); hold on
end
draw_circle(p_o, r_o); hold on
plot(x_d(1), x_d(2), 'r.', MarkerSize = 20); hold on
grid on
%xlim([lim_min, lim_max]);
%ylim([lim_min, lim_max]);
xlabel('p_x (m)')
ylabel('p_y (m)')

figure
for n = 1:N
    plot((0:size(h_hist{n},1)-1) * dt, h_hist{n}(:,1)); hold on
end
xlabel('Time (s)');
ylabel('CP-CBF: h(x_t)');
grid on

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
