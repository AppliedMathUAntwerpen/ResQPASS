close all; 
clear all; 

% Size of image
N = 20;

% Construction of the transformation
e = ones(N,1);
A1d = spdiags([e,e],[0,1],N,N);
A = [kron(A1d,speye(N)); kron(speye(N),A1d)];

% Image (two 2x3 white regions, rest black)
IM=zeros(N,N);
IM(4:6,6:7)=1;
IM(14:16,9:10)=1;

% Transform and add noise
b = A*IM(:);
e = 0.01*randn(length(b),1);
b = b + e;

% Naive solution
sol = A\b;

% Show A, original, transformed and naive solution
figure;
subplot(2,2,1)
spy(A)
title('A')

subplot(2,2,2)
imshow(IM, InitialMagnification=800)
title('Original (x)')

subplot(2,2,3)
imshow(reshape((b+min(b))/max(b+min(b)),N,2*N), InitialMagnification=800)
title('Ax+e = b (rescaled)')

subplot(2,2,4)
imshow(reshape(sol,N,N), InitialMagnification=800)
title('A\b')

%%  \|Ax-b\|^2 = (Ax-b)^T(Ax-b) = x' A'*A*x -2b'*Ax
%  l <= x <= u 
l = zeros(N*N,1);
u = ones(N*N,1);

% Bound constraint regularized solution
[sol2,fval,exitflag,output,lambda] = quadprog( A'*A, -A'*b,[],[],[],[],l,u);

% Figure comparing this solution to its lagrange multipliers and the
% other solutions obtained so far
figure;
subplot(2,2,1)
imshow(reshape(sol2,N,N));
title('Solution with l<x<u constraint')

subplot(2,2,3)
imshow(reshape(lambda.lower/max(lambda.lower),N,N))
title('\lambda rescaled')

subplot(2,2,4)
imshow(reshape(lambda.upper/max(lambda.upper),N,N))
title('\mu rescaled')

subplot(2,2,2)
hold all
plot(IM(:));
plot(sol)
plot(sol2)
legend('x', 'A\b', 'l<x<u')

%% Wolfe dual problem 
% 1/2 (\lambda-\mu)^T H (\lambda-\mu) +b^T A H(\lambda-\mu) -l^T \lambda  + u^T \mu
% lambda >= 0 
% mu >= 0 

% solution is l + x

H = inv(A'*A);

[sol3,fval,exitflag,output,lambda2] = quadprog([H -H; -H H], [H*A'*b-l ;-H*A'*b+u],[],[],[],[],zeros(2*N*N));

figure
plot(sol3);
title('solution of dual')

x = lambda2.lower(1:N*N);
y = lambda2.lower(N*N+1:end);
sol4 = l+x;

figure;
sgtitle('primal vs dual')
subplot(1,2,1)
hold all;
plot(x)
plot(N*N+1:2*N*N, y)
plot(sol2, 'x')
legend('x', 'y', 'primal sol')

subplot(1,2,2)
plot(sol4,'x'); hold all; plot(sol2)
legend('dual sol', 'primal sol')

%% Projected Dual problem. 
% 1/2 (\lambda-\mu)^T V^T H V (\lambda-\mu) +b^T A H V (\lambda-\mu) -l^T V \lambda  + u^T V \mu
%  V*lambda >= 0 
%  V*mu >= 0
% res =  A'*(Ax-b) - \lambda + \mu


v1 = A'*(A*l-b); % Initial "residual"
V = [v1/norm(v1) zeros(N*N,1) ; 
     zeros(N*N,1) v1/norm(v1)];% Basis

for iter=1:15
    disp(iter)
    %Solve the projected problem
    [coeff,fval(iter),exitflag,output,lambda3] = quadprog(V'*[H -H; -H H]*V, ...
    V'*[(H*A'*b-l);(-H*A'*b+u)],-V,zeros(2*N*N,1));
    
    %Dual solution
    x = l + lambda3.ineqlin(1:N*N);
    
    %Residual
    res = A'*(A*x-b) - V(1:N*N,1:2:2*iter)*coeff(1:2:2*iter) + V(1:N*N,2:2:2*iter)*coeff(2:2:2*iter);
    residu(iter)= norm(res);

    %Objective
    obj(iter) = norm(A*x-b);
    
    %Expansion of the basis
    V = [V [res/norm(res) zeros(N*N,1); zeros(N*N,1) res/norm(res)]];
    if obj(iter) < 2*norm(e)
        break
    end
end
figure; 
subplot(2,2,1)
plot(res); title('final residual');
subplot(2,2,2)
imshow(reshape(x,N,N)); title('Dual solution');
subplot(2,2,3) 
plot(x,'x');hold all; plot(sol); legend('Dual solution', 'A\b')
subplot(2,2,4)
semilogy(residu); 
hold all; 
semilogy(obj); 
% plot(log10(fval)); 
semilogy(0*fval + norm(e))
legend('residu', 'obj', 'noise')




