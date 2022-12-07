set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

%%
close all; clear all

rng('default')

% sizes
M = 1000;
N = 600;

%percentage of ones in A
percNZ = 4;

% projection matrix
A = randi(100,M,N);
A = (A<=4);
A = sparse(A);

% Creation of initial state
% Random values between -1, 1
x = 2*(randi(2,N,1)-1)-1;

% Some limited number is equal to 0 
% x(randi(N,N-50,1)) = 0;
x(randperm(numel(x),N/2)) = 0;

b = A*x + randn(M,1)*10e-3;

sigma = norm(b-A*x)^2;
x_exact = x;

%For "exact" solution
H = A'*A;
f = -b'*A;

% x0 = x + 0.5*randn(N,1);
% ct = norm(x_exact)^2;
%%
i=0;
mVec = [0,1,2,4,8,16,32,64,128];
% mVec = 1;
for m = mVec
i = i+1;

l = -1e6*ones(N,1);
u =  1e6*ones(N,1);
l(1:m) = -0.5*abs(x_exact(1:m))-1e-2;
u(1:m) =  0.5*abs(x_exact(1:m))+1e-2;
% l(1:m) = -ones(m,1);
% u(1:m) =  ones(m,1);

%comment to enable limited bounds
% l = -ones(N,1);
% u =  ones(N,1);
%%  exact solution


sol = quadprog(H,f,[],[],[],[], l, u);

exact_obj(i) = (A*sol-b)'*(A*sol-b);
%% Approximate solution
[y,V,x,obj{i},res{i},WS,~,LAM,MU] = subspace_qpas_restarted_krylov_functie(A,b,l,u);
% [y,V,x,obj,res,WS,nIters,LAM,MU] = subspace_qpas_restarted_krylov_functie(A,b,l,u);


end

%%

figure;
subplot(1,2,1)
for i = 1:length(mVec)
    semilogy(res{i})
%     semilogy(res)
    hold on;
    xlabel('Iteration')
end
title('Norm of residual')
leg = legend(num2str(mVec'));
title(leg,'$m_{\max}=$')

% figure;
subplot(1,2,2)
for i = 1:length(mVec)
    semilogy(abs(obj{i}-exact_obj(i)))
%     semilogy(abs(obj-exact_obj))
    hold on;
    xlabel('Iteration')
end
title('Error objective')
leg = legend(num2str(mVec'));
title(leg,'$m_{\max}=$')

%% Warm start behaviour
exact_obj = exact_obj(end);

tic
[y_ws,V_ws,x_ws,obj_ws,res_ws,WS_ws,nIters_ws,LAM_ws,MU_ws] = subspace_qpas_restarted_krylov_functie(A,b,l,u,300);
time_ws = toc;
tic
[y_nws,V_nws,x_nws,obj_nws,res_nws,WS_nws,nIters_nws,LAM_nws,MU_nws] = subspace_qpas_restarted_krylov_functie(A,b,l,u,300,false);
time_nws = toc;

%%
figure;
hold on
plot(cellfun(@length,WS_ws))
plot(cellfun(@length,WS_nws))

figure;
semilogy(abs(obj_ws-exact_obj))
hold on
semilogy(abs(obj_nws-exact_obj))

figure;
subplot(1,2,1);
hold on
plot(nIters_ws)
plot(nIters_nws)
legend(['Warm-start: ',num2str(time_ws,'%.2f'), 's'],['No warm-start: ',num2str(time_nws,'%.2f'), 's'])
xlim([0,max(length(nIters_ws), length(nIters_nws))])
xlabel('$k$')
ylabel(["\# QPAS"; "Iterations"], Rotation=0, HorizontalAlignment='right')
title('Number of inner iterations', 'FontSize',12)

subplot(1,2,2);
hold on
plot(cumsum(nIters_ws))
plot(cumsum(nIters_nws))
legend(['Warm-start: ',num2str(time_ws,'%.2f'), 's'],['No warm-start: ',num2str(time_nws,'%.2f'), 's'])
xlim([0,max(length(nIters_ws), length(nIters_nws))])
xlabel('$k$')
ylabel(["Total"; "QPAS"; "Iterations"], Rotation=0, HorizontalAlignment='right')
title('Total number of inner iterations', 'FontSize',12)

%% Limitting Inner Itterations
innerLimits = [5,10,25,50,100,200,300];
for i = innerLimits
tic
[y_lim{i},V_lim{i},x_lim{i},obj_lim{i},res_lim{i},WS_lim{i},nIters_lim{i},LAM_lim{i},MU_lim{i}] = subspace_qpas_restarted_krylov_functie(A,b,l,u,i);
time{i} = toc;
end

%%
% legendString = [];
figure;
subplot(1,2,1)
hold on
k=1;
for i = innerLimits
    plot(nIters_lim{i})
    legendString(k) = strcat(num2str(i), ": ", num2str(time{i}, '%.2f'), "s");
    k=k+1;
end
legend(legendString)
xlabel('$k$')
ylabel(["\# QPAS", "Iterations"], Rotation=0, HorizontalAlignment="right")
title("Inner iterations")

subplot(1,2,2)
for i = innerLimits
    semilogy(abs(obj_lim{i}-exact_obj))
    hold on
end
legend(legendString)
xlabel('$k$')
ylabel("$|f(V_ky_k)-f(x^*)|$", Rotation=0, HorizontalAlignment="right")
title("Error Objective")

figure;
for i = innerLimits
    plot(cumsum(nIters_lim{i}))
    hold on
end
legend(legendString)

%%
% h=figure;
% for i=1:nIters 
%     subplot(1,2,1);
%     plot(abs(V(:,i+1)));
%     hold on;
%     plot(LAM(:,i));
%     plot(MU(:,i));
%     hold off;
%     ylim([0 2]);
%     legend('$r_k$', '$\lambda$', '$\mu$')
% 
%     subplot(1,2,2); 
%     semilogy(res(1:i));
%     hold on; 
%     semilogy(abs(obj(1:i)-exact_obj)); 
%     hold off; 
%     title(i);
%     xlim([1,nIters])
%     ylim([1e-14, 1e5])
%     legend('$\|r_k\|$', 'Error objective')
% %     pause(0.1); 
%     print(sprintf('animate/spiky%d.eps',i),'-depsc2'); 
% end