%% Initialise Problem
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
m=128;
    l = -Inf*ones(N,1);
    u =  Inf*ones(N,1);
    l(1:m) = -0.5*abs(x(1:m))-1e-2;   % Enforce a bound on some indices
    u(1:m) =  0.5*abs(x(1:m))+1e-2;

%% Experiment    
tic
[xWarm,yWarm,VWarm,resWarm,itersWarm] = ResQPASS(A,b,l,u,1000);
timeWarm = toc;
tic
[xCold,yCold,VCold,resCold,itersCold] = ResQPASS(A,b,l,u,1000,[],min(M,N),false);
timeCold = toc;

%% Figure
width = 17; height = 7;
markings = {'-o', '-square','-diamond','-^','-*','-pentagram'};
colors = {'#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E','#E6AB02'};

figure(Units="centimeters", Position=[4 4 width height], PaperUnits="centimeters", PaperSize=[width height]);
subplot(1,2,1)
plot(itersCold,markings{1},Color=colors{1},MarkerIndices=1:10:length(itersCold))
hold on
plot(itersWarm,markings{4},Color=colors{4},MarkerIndices=1:10:length(itersWarm))
xlabel("Iteration $k$")
ylabel(["\# QPAS"; "iterations"], Rotation=0, HorizontalAlignment='right')
leg=legend(strcat("No warm-start: ", num2str(timeCold,'%.2f'),"s"),strcat("Warm-start: ", num2str(timeWarm,'%.2f'),"s"));
leg.BoxFace.ColorType='truecoloralpha';
leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');

subplot(1,2,2)
plot(cumsum(itersCold),markings{1},Color=colors{1},MarkerIndices=1:10:length(itersCold))
hold on
plot(cumsum(itersWarm),markings{4},Color=colors{4},MarkerIndices=1:10:length(itersWarm))
xlabel("Iteration $k$")
ylabel(["Total";"\# QPAS"; "iterations"], Rotation=0, HorizontalAlignment='right')
leg=legend(strcat("No warm-start: ", num2str(timeCold,'%.2f'),"s"), ...
    strcat("Warm-start: ", num2str(timeWarm,'%.2f'),"s"));
leg.BoxFace.ColorType='truecoloralpha';
leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');