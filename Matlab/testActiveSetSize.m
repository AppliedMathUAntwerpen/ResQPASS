% Script to tetst the effect of the size of the active set on the ResQPASS
% method.

%% Initialise Problem
rng('default') %Makes experiments repeatable

% Make a matrix A with 4% 1's and 96% 0's
M = 1000; N = 600;
densityNZ = 0.04;
A = (sprand(M,N,densityNZ)>0);

% Unconstrained solution has entries 0 (50%) or ±1 (%50)
x = 2*(randi(2,N,1)-1)-1;
x(randperm(numel(x),N/2)) = 0;

% RHS
b = A*x;

% For comparison with quadprog
H = A'*A;
f = -b'*A;

%% Experiment
i=0;
mVec = [0,8,16,32,64,128];
for m = mVec
    i = i+1;
    l = -Inf*ones(N,1);
    u =  Inf*ones(N,1);
    l(1:m) = -0.1*abs(x(1:m))-1e-2;   % Enforce a bound on some indices
    u(1:m) =  0.1*abs(x(1:m))+1e-2;

    % ResQPASS solution comparison for all versions
    tic
    [x_res{i},y,~,res{i},~,obj{i}] = ResQPASSv3(A,b,l,u,10);
    v3(i) = toc;
    tic
    [x_res{i},y,~,res{i},~,obj{i}] = ResQPASSv2(A,b,l,u,10);
    v2(i) = toc;
    tic
    [x_res{i},y,~,res{i},~,obj{i}] = ResQPASS(A,b,l,u,10);
    v1(i) = toc;

    % quadprog solution
    options = optimoptions(@quadprog,'Algorithm', 'active-set','Display','off','OptimalityTolerance',1e-15);
    [x_quad{i},~,flag_quad] = quadprog(H,f,[],[],[],[], l, u, zeros(size(x)), options);
    if flag_quad < 1
        warning("quadprog exited with flag %i", flag_quad)
    end
end

%% Figures
width = 17; height = 7;


markings = {'-o', '-square','-diamond','-^','-*','-pentagram'};
colors = {'#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E','#E6AB02'};

figure(Units="centimeters", Position=[4 4 width height], PaperUnits="centimeters", PaperSize=[width height]);
for i = 1:length(mVec)
    % Plot residual
    subplot(1,2,1)
    semilogy(res{i},markings{i},Color=colors{i},MarkerIndices=1:10:length(res{i}))
    hold on;
    xlabel('Iteration $k$')
    ylabel('$\|r_k\|$', Rotation=0, HorizontalAlignment='right')
    title("Norm of residual")
    ylim([1e-9 1e5])
    
    % Plot error of the objective between ResQPASS and quadprog
    subplot(1,2,2)
    errorObjective = abs(norm(A*x_quad{i}-b)^2 - (obj{i}));
    errorObjective(errorObjective==0) = eps;
    semilogy(errorObjective, markings{i},Color=colors{i},MarkerIndices=1:10:length(res{i}))
    hold on;
    xlabel("Iteration $k$")
    ylabel("$\Big|\|Ax_k-b\|_2^2-\|A\tilde{x}-b\|_2^2\Big|$")
    title("Error of objective")
    ylim([1e-16 1e5])
end

% Add legends
subplot(1,2,1)
leg = legend(num2str(mVec'));
title(leg,'$m_{\max}=$')
leg.BoxFace.ColorType='truecoloralpha';
leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');

subplot(1,2,2)
leg = legend(num2str(mVec'));
title(leg,'$m_{\max}=$')
leg.BoxFace.ColorType='truecoloralpha';
leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');