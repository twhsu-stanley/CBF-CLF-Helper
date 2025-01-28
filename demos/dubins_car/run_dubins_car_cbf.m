clc
clear
close all

dt = 0.01;
T = 6.5;
N = 10; % number of paths
x0 = [0; 5; 0]; % initial state
x_d = [10, 2, 0]; % desired state

params.v = 2; % velocity

% Obstacle position
params.xo = 5;
params.yo = 4;
% Obstacle radius
params.d = 2;
params.cbf_gamma0 = 15;

% Desired target point
params.xd = x_d(1);
params.yd = x_d(2);

%params.clf.rate = 1;
params.weight.slack = 20;

params.cbf.rate = 1;

dubins = DubinsCar(params);

dyn_fg = @dubins.dynamics;
controller_nominal = @dubins.ctrlNominal;
Kp = 1;%0.7
controller_cbf = @dubins.ctrlCbfQp;
controller_scbf = @dubins.ctrlStochasticCbfQp;
%controller_clfcbf = @dubins.ctrlCbfClfQp;

tt = 0:dt:T;
% initialize traces.
xs = zeros(N, length(tt), dubins.xdim);
us = zeros(N, length(tt)-1);
hs = zeros(N, length(tt)-1);

sigma = [0.1; 0.1; 0.1] * 0;

for n = 1:N
    for k = 1:length(tt)-1
        if k == 1 
            xs(n, 1, :) = [0; 2 + 6 *rand; -pi + 2*pi*rand]; % initial state;
        end

        t = tt(k);
        x = squeeze(xs(n, k, :));
        % Determine control input
        u_ref = controller_nominal(x, x_d, Kp);
        %[u, slack, h, V] = controller_clfcbf(x,u_ref);
        %[u, slack, h, V] = controller_clfcbf(x, sigma, u_ref);
        %[u, h, feas, comp_time] = controller_cbf(x, u_ref);
        [u, h, feas, comp_time] = controller_scbf(x, u_ref, sigma);
        if feas == 0
            continue
        end

        us(n, k) = u;
        hs(n, k) = h;

        % Run one time step propagation.
        xs(n, k+1, :) = x + dyn_fg(t, x, u) * dt + sqrt(dt) * sigma * randn;
    end
end


%% Plotting
p_o = [params.xo; params.yo];
r_o = params.d;

figure
subplot(3,1,1)
plot(tt, squeeze(xs(:,:,1)))
xlabel('Time (s)');
ylabel('p_x (m)')
grid on

subplot(3,1,2)
plot(tt, squeeze(xs(:,:,2)))
xlabel('Time (s)');
ylabel('p_y (m)')
grid on

subplot(3,1,3)
plot(tt, squeeze(xs(:,:,3)))
xlabel('Time (s)');
ylabel('theta (rad)')
grid on

%figure
%plot(tt(1:end-1), us)
%xlabel('t')
%ylabel('u [rad/s]')
%grid on

figure
for n = 1:N
    plot(xs(n,:, 1), xs(n,:, 2)); hold on
end
draw_circle(p_o, r_o);
grid on
%xlim([lim_min, lim_max]);
%ylim([lim_min, lim_max]);
xlabel('p_x (m)')
ylabel('p_y (m)')

figure
plot(tt(1:end-1), hs)
xlabel('Time (s)');
ylabel('SZCBF: h(x_t)');
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