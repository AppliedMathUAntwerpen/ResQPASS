%% Timings
rng('default') %Makes experiments repeatable

%Size of A: MxN
M = 10000;
N = 6000;

%Percentage of ones in A
percNZ = 0.04;

%Projection matrix
A = sprandn(M,N,percNZ); %% TODO >0 of niet

% Creation of initial state
% Random values between -1, 1
x = 2*(randi(2,N,1)-1)-1;

% Some limited number is equal to 0
x(randperm(numel(x),N/2)) = 0;

%Right hand
b = A*x;

%For "exact" solution
H = A'*A;
f = -b'*A;

for i=1:120 
    l = -1e6*ones(N,1);
    u =  1e6*ones(N,1);
    l(1:i) = -0.5*abs(x(1:i))-1e-2;
    u(1:i) =  0.5*abs(x(1:i))+1e-2;

    if(i<=20)
        tic
        [solAS,~] = quadprog(H,f,[],[],[],[], l, u,0*l,optimoptions("quadprog","Algorithm","active-set"));
        tijdAS(i) = toc;
    end

    if(i<32)
        tic
        solQPAS= qpasNocedal(A'*A,f',[-eye(length(l)); eye(length(u))],[-l;u],0*l);
        tijdQPAS(i) = toc;

    end

    tic
    [solResQPASS,~,~,res] = ResQPASSv2(A,b,l,u);
    tijdResQPASS(i) = toc;

    tic
    [solIP,~] = quadprog(H,f,[],[],[],[], l, u);
    tijdIP(i) = toc;

    tic
    [solQPASSchur,ws,it]= qpasCholesky(full(chol(A'*A))',f',[-eye(length(l)); eye(length(u))],[-l;u],0*l,[],100);
    tijdQPASSchur(i) = toc;


end

%%
colors = {'#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E'};
markings = {'-o', '-square','-diamond','-^','-*','-pentagram','-+', '-x'};
figure(Units="centimeters", Position=[4 4 17 9], PaperUnits="centimeters", PaperSize=[17 9]);
plot(tijdResQPASS,markings{1}, Color=colors{1}, MarkerIndices=1:5:120)
hold on
plot(tijdAS(tijdAS<20), markings{2}, Color=colors{2}, MarkerIndices=1:5:120)
plot(tijdQPAS, markings{5},Color=colors{5}, MarkerIndices=1:5:120)
plot(tijdQPASSchur, markings{4}, Color=colors{4}, MarkerIndices=1:5:120)
plot(tijdIP, markings{3}, Color=colors{3}, MarkerIndices=1:5:120)

legend("ResQPASS", "QPAS (MATLAB)", "QPAS (Nocedal)", "QPAS (improved)", "IP (MATLAB)")
xlabel("$i_{\max}$")
ylabel("Time (s)")

