%% Initialisation
rng('default') %Makes experiments repeatable

% Make a matrix A with 4% 1's and 96% 0's
M = 1000; N = 600;
percNZ = 4;
A = randi(100,M,N);
A = (A<=percNZ);
A = sparse(A);

% Unconstrained solution has entries 0 (50%) or ±1 (%50)
x = 2*(randi(2,N,1)-1)-1;
x(randperm(numel(x),N/2)) = 0;

% RHS
b = A*x;

% For comparison with quadprog
H = A'*A;
f = -b'*A;

m=128;
    l = -Inf*ones(N,1);
    u =  Inf*ones(N,1);
    l(1:m) = -0.5*abs(x(1:m))-1e-2;   % Enforce a bound on some indices
    u(1:m) =  0.5*abs(x(1:m))+1e-2;

[x_quad,~,flag_quad] = quadprog(H,f,[],[],[],[], l, u, zeros(size(x)), optimset('Algorithm', 'active-set','Display','off'));

%% Experiment
innerIterations = [3 5 10 25 50 100];
timing = zeros(size(innerIterations));

width = 17; height = 7;
markings = {'-o', '-square','-diamond','-^','-*','-pentagram'};
colors = {'#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E','#E6AB02'};
legendString={};

figure(Units="centimeters", Position=[4 4 width height], PaperUnits="centimeters", PaperSize=[width height]);
% One loop for timing
for i=1:length(innerIterations)
tic
[x,y,V,res,iters] = ResQPASS(A,b,l,u,innerIterations(i));
timing = toc;
subplot(1,2,1);
plot(iters,markings{i},Color=colors{i},MarkerIndices=1:10:length(iters)); 
ylabel(["\# QPAS"; "iterations"], Rotation=0, HorizontalAlignment='right')
xlabel("Iteration $k$")
title("Inner iterations")
hold on;
legendString{i} = strcat(num2str(innerIterations(i)),": ",num2str(timing,'%.2f'),"s");
end
leg = legend(legendString);
title(leg,"Max inner iterations")
leg.BoxFace.ColorType='truecoloralpha';
leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');

% One loop for errors
for i=1:length(innerIterations)
[x,y,V,res,iters,obj] = ResQPASS(A,b,l,u,innerIterations(i));
errorObjective = abs(norm(A*x_quad-b)^2 - obj);
errorObjective(errorObjective==0) = eps;
subplot(1,2,2)
semilogy(errorObjective,markings{i},Color=colors{i},MarkerIndices=1:10:length(iters)); 
ylabel("$\Big|\|Ax_k-b\|_2^2-\|Ax^*-b\|_2^2\Big|$")
xlabel("Iteration $k$")
ylim([1e-16,1e5])
title("Error objective")
hold on;
end
leg = legend(legendString,Location="southwest");
title(leg,"Max inner iterations")
leg.BoxFace.ColorType='truecoloralpha';
leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
