FWETBlue = "#006CA9";
UABlue = "#002E65";
markings = {'-o', '-square','-diamond','-^','-*','-pentagram'};
colors = {'#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E','#E6AB02'};
%%
rng('default') %Makes experiments repeatable
% MM = [600,1000];
% NN = [1000,600];
MM = [1800,3000];
NN = [3000,1800];
nRuns = 15;

time_cg = zeros(nRuns,length(MM));
time_pcg = time_cg;
time_ResQPASS = time_cg;
time_ResQPASS_dense = time_cg;

resvec_cg = cell(nRuns,length(MM));
resvec_pcg = resvec_cg;
resvec_ResQPASS = resvec_cg;
resvec_ResQPASS_dense = resvec_cg;

tol = 1e-8;
maxit = 200;

for j=1:length(MM)

for i = 1:nRuns
    i
    M = MM(j)*i;
    N = NN(j)*i;
    A = randn(M,N);
    A(A<0) = 0;
    A(A>0.1) = 0;
    A = sparse(A);
    x_ex = randn(N,1);
    b = A*x_ex;

    tic
    % CG applied to the normal equations (A'*A is dense)
    [~,~,~,~,resvec_pcg{i,j}] = lsqr((A),b,tol,maxit);
    time_pcg(i,j) = toc;

%     tic
%     % Improved CG for normal equations
%     [~,~,~,~,resvec_cg{i,j}] = cg_AtA(A,A'*b,zeros(size(x_ex)), tol,maxit);
%     time_cg(i,j) = toc;

    tic
    % ResQPASS
    [~,~,~,resvec_ResQPASS{i,j}] = ResQPASSv2((A),b);
    time_ResQPASS(i,j) = toc;

%     tic
%     % ResQPASS with dense matrix (to compare with dense PCG)
%     [~,~,~,resvec_ResQPASS_dense{i,j}] = ResQPASS(full(A),b);
%     time_ResQPASS_dense(i,j) = toc;
end
end


%%
figure(Units="centimeters", Position=[4 4 17 7], PaperUnits="centimeters", PaperSize=[17 7]);
for j = 1:2
colors = {'#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E'};
figure(1)
subplot(1,2,j)
hold on
plot(MM(j)*(1:nRuns),time_ResQPASS(:,j),'-diamond',Color=colors{4})
plot(MM(j)*(1:nRuns),time_pcg(:,j),'-o',Color=colors{1})
xlabel("Height $m$ ")
ylabel("Time (s)")
legend("ResQPASS","LSQR (MATLAB)",Location="northwest")
if(MM(j)>NN(j))
    title("Overdetermined (10:6)")
else
    title("Underdetermined (6:10)")
end
ax = gca;
ax.Box = 'on';
end