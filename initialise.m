clc, clear, close all, 

%% Initialise problem
n = 32;
[A,~,x_exact] = PRblur(n, PRset('trueImage', 'hst', 'BlurLevel','mild','PSF', 'rotation'));
% A = sparse(full(A));
A = A/sum(A(1,:));
b_exact = A*x_exact;
b = b_exact + 0.01*randn(size(b_exact)) - 0.5;

figure;
imshow(reshape(b+0.5,n,n),'InitialMagnification', 800)
delta = 0.5;
N = n^2;

% A = randn(500,500);
% b = randn(500,1);
% N=500;

% H = A'*A;
% f = -b'*A;
% sol = quadprog(H,f,[],[],[],[],-delta*ones(N,1),delta*ones(N,1));
% % sol = qpas_schur(H,f',[eye(N); -eye(N)], delta*[ones(N,1); ones(N,1)], zeros(size(b)), []);
% exact_obj = (A*sol-b)'*(A*sol-b);

% figure;
% imshow(reshape(sol+0.5,n,n))

%%
u = delta*ones(N,1);
l=-u;
