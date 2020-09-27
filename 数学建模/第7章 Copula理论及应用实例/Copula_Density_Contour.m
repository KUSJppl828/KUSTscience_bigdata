[U,V] = meshgrid(linspace(0.01,0.99,30));
Y = copulapdf('Gaussian',[U(:), V(:)],0.7);
surf(U,V,reshape(Y,size(U)))
xlabel('u')
ylabel('v')
zlabel('Gaussian c(u,v)')
figure
[C,h] = contour(U,V,reshape(Y,size(U)),50);
set(h,'color','k')
clabel(C,h,'manual')
xlabel('u')
ylabel('v')

[U,V] = meshgrid(linspace(0.01,0.99,30));
Y = copulapdf('t',[U(:), V(:)],0.7,5);
surf(U,V,reshape(Y,size(U)))
xlabel('u')
ylabel('v')
zlabel('t c(u,v)')
figure
[C,h] = contour(U,V,reshape(Y,size(U)),50);
set(h,'color','k')
clabel(C,h,'manual')
xlabel('u')
ylabel('v')

[U,V] = meshgrid(linspace(0.01,0.99,30));
Y = copulapdf('Gumbel',[U(:), V(:)],1.5);
surf(U,V,reshape(Y,size(U)))
xlabel('u')
ylabel('v')
zlabel('Gumbel c(u,v)')
figure
[C,h] = contour(U,V,reshape(Y,size(U)),60);
set(h,'color','k')
clabel(C,h,'manual')
xlabel('u')
ylabel('v')

[U,V] = meshgrid(linspace(0.01,0.99,30));
Y = copulapdf('Clayton',[U(:), V(:)],1);
surf(U,V,reshape(Y,size(U)))
xlabel('u')
ylabel('v')
zlabel('Clayton c(u,v)')
figure
[C,h] = contour(U,V,reshape(Y,size(U)),60);
set(h,'color','k')
clabel(C,h,'manual')
xlabel('u')
ylabel('v')

[U,V] = meshgrid(linspace(0.01,0.99,30));
Y = copulapdf('Frank',[U(:), V(:)],2);
surf(U,V,reshape(Y,size(U)))
xlabel('u')
ylabel('v')
zlabel('Frank c(u,v)')
figure
[C,h] = contour(U,V,reshape(Y,size(U)),10);
set(h,'color','k')
clabel(C,h,'manual')
xlabel('u')
ylabel('v')