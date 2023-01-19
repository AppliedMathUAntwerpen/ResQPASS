close all; 
clear all; 
N = 20
e = ones(N,1)
A1d = spdiags([e,e],[0,1],N,N)
A= [kron(A1d,speye(N)); kron(speye(N),A1d)]

IM=zeros(N,N)
IM(4:6,6:7)=1

IM(14:16,9:10)=1

imshow(IM)

b = A*IM(:)
e =  0.01*randn(length(b),1)
b = b +e

sol =    A\b

imshow(reshape(sol,N,N))

%%  \|Ax-b\|^2 = (Ax-b)^T(Ax-b) = x' A'*A*x -2b'*Ax
%  l <= x <= u 
l = zeros(N*N,1)
u = ones(N*N,1)

[sol2,fval,exitflag,output,lambda] = quadprog( A'*A, -A'*b,[],[],[],[],l,u);
figure;
imshow(reshape(sol2,N,N));
figure; imshow(reshape(lambda.lower,N,N))

figure; imshow(reshape(lambda.upper,N,N))

%%
figure; hold all
plot(IM(:));
plot(sol)
plot(sol2)

%% dual problem 
%   1/2 (\lambda-\mu)^T H (\lambda-\mu) +b^T A H(\lambda-\mu) -l^T \lambda  + u^T \mu
% lambda >= 0 
% mu >= 0 

% solution is l + x

H = inv(A'*A);

[sol3,fval,exitflag,output,lambda2] = quadprog([H -H; -H H], [H*A'*b-l ;-H*A'*b+u],[],[],[],[],zeros(2*N*N))


figure
plot(sol3);
x = lambda2.lower(1:N*N)
y = lambda2.lower(N*N+1:end)
figure; plot(lambda2.lower); hold all; plot(sol2)
sol4 = l+x

figure; plot(sol4,'x'); hold all; plot(sol2)

%% Projected Dual problem. 
% 1/2 (\lambda-\mu)^T V^T H V (\lambda-\mu) +b^T A H V (\lambda-\mu) -l^T V \lambda  + u^T V \mu
%  V*lambda >= 0 
%  V*mu >= 0
% res =  A'*(Ax-b) - \lambda + \mu


v1 = A'*(A*l-b)
V = [v1/norm(v1) zeros(N*N,1) ; 
     zeros(N*N,1) v1/norm(v1)]
for iter=1:15
    [coeff,fval(iter),exitflag,output,lambda3] = quadprog(V'*[H -H; -H H]*V, ...
    V'*[(H*A'*b-l);(-H*A'*b+u)],[-V],zeros(2*N*N,1))
    x = l + lambda3.ineqlin(1:N*N)
    res = A'*(A*x-b) - V(1:N*N,1:2:2*iter)*coeff(1:2:2*iter) + V(1:N*N,2:2:2*iter)*coeff(2:2:2*iter)
    residu(iter)= norm(res)
    obj(iter) = norm(A*x-b)
    V = [V [res/norm(res) zeros(N*N,1); zeros(N*N,1) res/norm(res)]]
    if obj(iter) < 2*norm(e)
        break
    end
end
figure; plot(res)
figure; imshow(reshape(x,N,N),'InitialMagnification',800)
figure; plot(x,'x');hold all; plot(sol)
 
figure; plot(log10(residu)); hold all; plot(log10(obj)); plot(log10(fval)); plot(0*fval + log10(norm(e)))





