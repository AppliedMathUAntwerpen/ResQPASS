%% Instability QPAS
rng('default') %Makes experiments repeatable

%Size of A: MxN
M = 10000;
N = 6000;

%Percentage of ones in A
percNZ = 4;

%Projection matrix
A = randi(100,M,N);
A = (A<=percNZ);
A = sparse(A);

% Creation of initial state
% Random values between -1, 1
x = 2*(randi(2,N,1)-1)-1;

% Some limited number is equal to 0 
x(randperm(numel(x),N/2)) = 0;
x_exact = x;

%Right hand
b = A*x; 

m = 256;

l = -1e6*ones(N,1);
u =  1e6*ones(N,1);
l(1:m) = -0.5*abs(x_exact(1:m))-1e-2;
u(1:m) =  0.5*abs(x_exact(1:m))+1e-2;

%Use a slightly reduced tollerance 

[x,y,V,~,~,~,recursiveError5] = ResQPASS(A,b,l,u,5,[],min(M,N),true,true,1e-7);
tic
[x,y,V,resRec5] = ResQPASS(A,b,l,u,5,[],min(M,N),true,true,1e-7);
timeRec5=toc;

[x,y,V,~,~,~,recursiveError10] = ResQPASS(A,b,l,u,10,[],min(M,N),true,true,1e-7);
tic
[x,y,V,resRec10] = ResQPASS(A,b,l,u,10,[],min(M,N),true,true,1e-7);
timeRec10=toc;

tic
[x,y,V,resDirect] = ResQPASS(A,b,l,u,10,[],min(M,N),true,false,1e-7);
timeDirect=toc;

%%
width = 17; height = 7;
markings = {'-o', '-square','-diamond','-^','-*','-pentagram'};
colors = {'#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E','#E6AB02'};

figure(Units="centimeters", Position=[4 4 width height], PaperUnits="centimeters", PaperSize=[width height]);
subplot(1,2,1)
semilogy(resDirect, markings{1}, Color=colors{1}, MarkerIndices=1:20:length(resDirect))
hold on
semilogy(resRec5, markings{2}, Color=colors{2}, MarkerIndices=1:20:length(resRec5))
semilogy(resRec10, markings{3}, Color=colors{3}, MarkerIndices=10:20:length(resRec10))
xlabel('Outer iteration $i$')
ylabel('$\|r_i\|$', Rotation=0, HorizontalAlignment='right')
leg=legend(strcat("Direct (10): ", num2str(timeDirect,'%.2f'), 's'), ...
    strcat("Recursive (5): ", num2str(timeRec5,'%.2f'), 's'), ...
    strcat("Recursive (10): ", num2str(timeRec10,'%.2f'), 's'), ...
    Location="southwest");
leg.BoxFace.ColorType='truecoloralpha';
leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');

subplot(1,2,2)
semilogy(recursiveError5,'square', MarkerSize=3, Color=colors{2})
hold on
semilogy(recursiveError10,'diamond', MarkerSize=3, Color=colors{3})
xlabel('Inner iteration $k$')
ylabel('$\|(Cx_k)_{rec} - (Cx_k)_{ex}\|$')
legend('Recursive (5)', 'Recursive (10)', Location='southeast')