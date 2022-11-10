close all; clear all

% sizes
M = 1000;
N = 600;

% projection matrix
A = randi(50,M,N)-1;
A = (A<2);
A = sparse(A);

% Creation of initial state
% Random values between -1, 1
x = 2*(randi(2,N,1)-1)-1;

% Some limited number is equal to 0 
x(randi(N,N-50,1)) = 0;

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
for m = mVec
i = i+1;

l = -1e6*ones(N,1);
u =  1e6*ones(N,1);
l(1:m) = -0.5*abs(x_exact(1:m))-1e-2;
u(1:m) =  0.5*abs(x_exact(1:m))+1e-2;
% l(1:m) = -ones(m,1);
% u(1:m) =  ones(m,1);
 
% l = -ones(N,1);
% u =  ones(N,1);
%%  exact solution


sol = quadprog(H,f,[],[],[],[], l, u);

exact_obj(i) = (A*sol-b)'*(A*sol-b);
%% Approximate solution
[~,~,~,obj{i},res{i},~,~] = subspace_qpas_restarted_krylov_functie(A,b,l,u);

end

%%

figure;
for i = 1:length(mVec)
    semilogy(res{i})
    hold on;
end
title('residu')
leg = legend(num2str(mVec'));
title(leg,'m=')

figure;
for i = 1:length(mVec)
    semilogy(abs(obj{i}-exact_obj(i)))
    hold on;
end
title('error objectief')
leg = legend(num2str(mVec'));
title(leg,'m=')

%%
% figure;
% for i=1:nIters 
%     subplot(1,2,1);
%     plot(abs(V(:,i+1)));
%     hold on;
% %     plot(LAM(:,i));
% %     plot(MU(:,i));
%     hold off;
%     ylim([0 2]);
% 
%     subplot(1,2,2); 
%     semilogy(res(1:i));
%     hold on; 
%     semilogy(abs(obj(1:i)-exact_obj)); 
%     hold off; 
%     title(i);
%     xlim([1,nIters])
%     ylim([1e-14, 1e5])
%     pause(0.01); 
% end