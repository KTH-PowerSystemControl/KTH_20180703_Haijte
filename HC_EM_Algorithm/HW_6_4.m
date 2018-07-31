%%
%Analisis Inteligente de Señales y Sistemas
%File: Tarea 6 - Punto 4
%Date: 17042017 EM Algorithm

clc;
close all;
clear all;

%Datos basicos para el algoritmo EM
k=5;%Numero de clases
X_data = load('dataX.txt');%Carga de datos de X
m = size(X_data,1);%Numero de datos de entrenamiento
n = size(X_data,2);%Dimensiones de la funcion de densidad Gaussiana

%%Algorith 8.1: [Elements_Statistical_Learning-Hastie] - Page 275
%Paso 1: Inicializacion de medias (mu_X), varianzas (Q_X) y probabilidad mixta 
%-Hint: [Elements_Statistical_Learning-Hastie] - Page 274 - A good way ...

%-Media: elemento aleatorio dentro de X_data. fila_x: ubica fila al azar dentro de X_data
fila_x = sort(datasample(1:m,k,'Replace',false));
mu_X = X_data(fila_x,:);

%-Covarianza: las k covarianzas son la misma, promedio de la covarianza X_data
Q_X_data = cov(X_data);
for i = 1:k
    Q_X{i} = Q_X_data/k;
end

%-Probabilidad mixta (alpha_k): inicialmente igual para todos (likelihood)
alpha_k(1:k)=1/k;

%-Calcular J(mu_X,Q_X,alpha_k)
%-Hint: Diapositivas Reconocimiento Patrones II - Slide 88
J_log_history=[];%Registro historico de J
tmp_sum_1=0;
%--Sumatoria i=1 to m
for i = 1:m
    x_i = X_data(i,:);
    tmp_sum_2=0;
    %--Sumatoria k=1 to K
    for j=1:k
        %---PDF de cada dato para cada clase
        p_k = (1/sqrt((2.*pi).^n.*det(Q_X{j}))).*exp(-(1/2).*(X_data(i,:)-mu_X(j,:))*inv(Q_X{j})*(X_data(i,:)-mu_X(j,:))');
        tmp_sum_2 = tmp_sum_2 + alpha_k(j)*p_k;
    end
    tmp_sum_1 = tmp_sum_1 + log(tmp_sum_2);
end
J_log_new = tmp_sum_1;
J_log_history = [J_log_history J_log_new];


%Paso 2 y Paso 3: deben realizarse hasta alcanzar convergencia
%-Hint: es util definir un umbral de iteraciones (no converge)
Iteracion_th = 100;%Umbral de iteraciones
%-Hint: debe definirse un criterio o umbral de convergencia 
Convergencia_th = 10e-9;%Umbral de convergencia

%-Pasos 2 y 3 iterando hasta alcanzar convergencia o iteraciones
J_log_old = 0;
EM_on = true;
EM_iteracion = 0;
while EM_on
    EM_iteracion = EM_iteracion+1
    J_log_old = J_log_new;
    
    %--Paso 2: Paso E - computa las responsabilidades
    gamma_k(1:m,1:k)=0;
    for i = 1:m
        den_gamma_k=0;
        for j = 1:k
            %---PDF de cada dato para cada clase
            p_k = (1/sqrt((2.*pi).^n.*det(Q_X{j}))).*exp(-(1/2).*(X_data(i,:)-mu_X(j,:))*inv(Q_X{j})*(X_data(i,:)-mu_X(j,:))');
            gamma_k(i,j) = alpha_k(j)*p_k;
            den_gamma_k = den_gamma_k + gamma_k(i,j);
        end
        %---Normalizar
        for j = 1:k
            gamma_k(i,j)= gamma_k(i,j)/den_gamma_k;
        end
    end
    
    %--Paso 3: Paso M - computa las nuevas medias, covarianzas y alpha_k
    %---Nuevas medias (mu_k)
    mu_k(1:k,1:n) = 0;
    den_mu_k(1:k)=0;
    for i=1:m
        den_mu_k = den_mu_k + gamma_k(i,:);
    end

    for j=1:k
        num_mu_k(1:n)=0;
        for i=1:m
            num_mu_k = num_mu_k + gamma_k(i,j)*X_data(i,:);
        end
        mu_k(j,:) = num_mu_k/den_mu_k(j);
    end
    
    %---Nuevas covarianzas (C_k)
    for j = 1:k
        num_C_k(1:n,1:n) = 0;
        den_C_k = 0;
        for i=1:m
            den_C_k = den_C_k + gamma_k(i,j);
            num_C_k = num_C_k + gamma_k(i,j)*((X_data(i,:)-mu_X(j,:))'*(X_data(i,:)-mu_X(j,:)));
        end
        C_k{j} = num_C_k /den_C_k;
    end
    
    %Importante: actualizar medias y covarianzas
    mu_X = mu_k;
    Q_X = C_k;
    
    %---Nuevos likelihood (alpha_k)
    alpha_k(1:k) = 0; 
    for i=1:m
        alpha_k = alpha_k + gamma_k(i,:);
    end
    alpha_k = alpha_k/m;

    %---Calcular nuevo J(mu_k,C_k,alpha_k)
    tmp_sum_1=0;
    %----Sumatoria i=1 to m
    for i = 1:m
        x_i = X_data(i,:);
        tmp_sum_2=0;
        %----Sumatoria k=1 to K
        for j=1:k
            %---PDF de cada dato para cada clase
            p_k = (1/sqrt((2.*pi).^n.*det(C_k{j}))).*exp(-(1/2).*(X_data(i,:)-mu_k(j,:))*inv(C_k{j})*(X_data(i,:)-mu_k(j,:))');
            tmp_sum_2 = tmp_sum_2 + alpha_k(j)*p_k;
        end
        tmp_sum_1 = tmp_sum_1 + log(tmp_sum_2);
    end
    J_log_new = tmp_sum_1;
    J_log_history = [J_log_history J_log_new];
    
    convergencia_valor = abs((J_log_new - J_log_old)/J_log_old);
    if(EM_iteracion >= Iteracion_th || convergencia_valor<Convergencia_th)
        EM_on = false;
    end
end

%Separacion en k Clases
clase_k = zeros(m,1);
for i=1:m
    max_gamma_k = max(gamma_k(i,:));
    clase_k(i) = find(gamma_k(i,:) == max_gamma_k);
end
K_data_1 = [];
K_data_2 = [];
for i=1:m
    if clase_k(i) == 1
        K_data_1 = [K_data_1; X_data(i,:)];
    else
        K_data_2 = [K_data_2; X_data(i,:)];
    end
end




%Grafica: razon de convergencia de EM
figure('Color',[1,1,1],'Position',[200 400 1200 600])
plot(1:size(J_log_history,2),J_log_history);
title('Algoritmo EM - Razon de convergencia')
xlabel('Iteraciones')
ylabel('J(\theta,\alpha)')

%Grafica: datos con identificacion de clases
figure('Color',[1,1,1],'Position',[200 400 1200 600])
plot(K_data_1(:,1),K_data_1(:,2),'*m')
hold on
plot(K_data_2(:,1),K_data_2(:,2),'*b')
hold on
plot(mu_k(1,1),mu_k(1,2),'d','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',10)
hold on
plot(mu_k(2,1),mu_k(2,2),'d','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10)
title('Algoritmo EM - Clasificador')
legend('Datos Clase_1','Datos Clase_2','Media Clase_1','Media Clase_2','Location','northwest')
xlabel('X_1')
ylabel('X_2')
%print -dpng HW_5_Punto_2_fig_LDA.png
