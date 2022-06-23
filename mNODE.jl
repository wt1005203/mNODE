## Libraries
begin
    ## install necessary packages if they are not installed yet according to "Project.toml" and "Manifest.toml"
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()

    ## load necessary packages
    println("Loading Modules")
    using Flux, DifferentialEquations, DiffEqFlux
    using Statistics, MLDataPattern, MultivariateStats
    using LinearAlgebra, Distances, Random
    using Distributions: Normal,Uniform
    using StatsBase: sample,Weights
    using Printf
    using Base.Iterators: partition
    using DelimitedFiles
    using Distributed
    using SharedArrays
    using MLDataUtils: kfolds
    println("Done")
end

begin
    path_prefix = "./processed_data/"
    ## Load Data
    X_train = readdlm(path_prefix * "X_train.csv",',',Float64,'\n')';
    y_train = readdlm(path_prefix * "y_train.csv",',',Float64,'\n')';
    X_test = readdlm(path_prefix * "X_test.csv",',',Float64,'\n')';
    y_test = readdlm(path_prefix * "y_test.csv",',',Float64,'\n')';
    compound_names = readdlm(path_prefix * "compound_names.csv",'\t');
    
    Nb = size(X_train)[1] # number of microbial/bacterial species
    Nr = size(y_train)[1] # number of metabolites
    println("="^50)
    println("The number of microbial taxa and the number of metabolites is:")
    println(Nb, " ", Nr)
    println("="^50)
end

## train mNODE
function mNODE_train(X_train, y_train, X_test, y_test, Nh, weight_decay)
    MLP1 = Chain(Dense(Nb,Nh,tanh))
    dudt = Chain(Dense(Nh,Nh,tanh),
                 Dense(Nh,Nh,tanh))
    MLP2 = Chain(Dense(Nh,Nr))

    n_ode = x->neural_ode(dudt,x,(0.0,1.0),Tsit5(),saveat=1.0)[:, end]
    model = x->MLP2(n_ode(MLP1(x)))

    function predict(θ,x)
        #node(x) = neural_ode(θ,x,(0.0,1.0),Tsit5(),saveat=1.0)[:, end ]
        q = hcat([ model(x[:,i]) for i in 1:size(x,2) ]...)
        return q
    end

    sqnorm(x) = sum(abs2, x)
    penalty() = sum(sqnorm, Flux.params(MLP1)) + sum(sqnorm, Flux.params(dudt)) + sum(sqnorm, Flux.params(MLP2))
    Loss(θ,x,y) = mean( [ Flux.mse( y[:,i], predict(θ,x[:,i]) ) for i in 1:size(x,2)] ) + weight_decay * penalty()

    ## Train
    ## Parameters
    N,NN = size(X_train);
    epochs = 100;
    early_stop = 20
    mb_size = 10;
    num_of_increase = 3

    ## Vars
    train_loss = []
    test_loss = []
    Qtst = []

    for tt in 1:1
        ## Callback function
        function cb()
            ltrn = Loss(model,X_train,y_train).data
            ltst = Loss(model,X_test,y_test).data

            push!(train_loss,ltrn)
            push!(test_loss,ltst)
        end

        train_error = Loss(model,X_train,y_train)
        test_error = Loss(model,X_test,y_test)
        @time for e in 1:epochs
            ## Weights for MLP, neural ODE, and MLP from inputs to outputs
            V1 = deepcopy(Flux.data.(params(MLP1)))
            V2 = deepcopy(Flux.data.(params(dudt)))
            V3 = deepcopy(Flux.data.(params(MLP2)))

            ## Inner loop
            learning_rate = 1e-2
            for mb in partition(randperm(size(X_train,2)), mb_size)
                l = Loss(model,X_train[:,mb],y_train[:,mb])
                Flux.back!(l)
                Flux.Optimise._update_params!(ADAM(learning_rate), params(MLP1))
                Flux.Optimise._update_params!(ADAM(learning_rate), params(dudt))
                Flux.Optimise._update_params!(ADAM(learning_rate), params(MLP2))
            end

            ## Reptile update
            for (w, v) in zip(params(MLP1),V1)
                dv = Flux.Optimise.apply!(ADAM(learning_rate), v, (w.data-v)/mb_size)
                @. w.data = v + dv
            end
            for (w, v) in zip(params(dudt),V2)
                dv = Flux.Optimise.apply!(ADAM(learning_rate), v, (w.data-v)/mb_size)
                @. w.data = v + dv
            end
            for (w, v) in zip(params(MLP2),V3)
                dv = Flux.Optimise.apply!(ADAM(learning_rate), v, (w.data-v)/mb_size)
                @. w.data = v + dv
            end

            train_error = Loss(model,X_train,y_train)
            test_error = Loss(model,X_test,y_test)

            # Report
            (e%1 == 0 || e==1) && cb()
            if e>early_stop
                if sum(((test_loss[end-early_stop+1:end] - test_loss[end-early_stop:end-1]).>0)) > early_stop/2+1
                    break
                end
            end
        end
    end

    ## generate predictions and calculate the prediction performance
    corr_test = zeros(size(y_test,1))
    y_test_pred = hcat([reshape(predict(model,X_test[:,i]), Nr) for i in 1:size(X_test, 2)]...)
    for i = 1:size(y_test,1)
        corr_test[i] = Flux.Tracker.data(cor(y_test_pred[i,:], y_test[i,:]))
    end
    println(mean(corr_test))
    return mean(corr_test)

end

## hyperparameter selection determined by the 5-fold cross validation on the training set
println("Begin the hyperparameter selection: ")
folds = kfolds((X_train, y_train), k = 5)
weight_decay_list = [1e-4, 1e-3, 1e-2]
Nh_list = [32, 64, 128]
Spearmac_CC_for_all_hyperparameters = zeros(size(weight_decay_list,1), size(Nh_list,1))
for i in 1:size(weight_decay_list,1)
    for j in 1:size(Nh_list,1)
        "#"^50 |> println
        weight_decay = weight_decay_list[i]
        Nh = Nh_list[j]
        println(weight_decay, " ", Nh)
        mean_spearman_cc_all = zeros(5)
        for k = 1:5
            ((X_train_5fold,y_train_5fold),(X_test_5fold,y_test_5fold)) = folds[k]
            mean_spearman_cc = mNODE_train(X_train_5fold, y_train_5fold, X_test_5fold, y_test_5fold, Nh, weight_decay)
            mean_spearman_cc_all[k] = mean_spearman_cc
        end
        println(mean(mean_spearman_cc_all))
        Spearmac_CC_for_all_hyperparameters[i,j] = mean(mean_spearman_cc_all)
    end
end

println(findmax(Spearmac_CC_for_all_hyperparameters)[1])
i1 = findmax(Spearmac_CC_for_all_hyperparameters)[2][1]
i2 = findmax(Spearmac_CC_for_all_hyperparameters)[2][2]
weight_decay = weight_decay_list[i1]
Nh = Nh_list[i2]
println("The selected hyperparameters are")
println(weight_decay, " ", Nh)

MLP1 = Chain(Dense(Nb,Nh,tanh))
dudt = Chain(Dense(Nh,Nh,tanh),
             Dense(Nh,Nh,tanh))
MLP2 = Chain(Dense(Nh,Nr))

n_ode = x->neural_ode(dudt,x,(0.0,1.0),Tsit5(),saveat=1.0)[:, end]
model = x->MLP2(n_ode(MLP1(x)))

function predict(θ,x)
    q = hcat([ model(x[:,i]) for i in 1:size(x,2) ]...)
    return q
end

sqnorm(x) = sum(abs2, x)
penalty() = sum(sqnorm, Flux.params(MLP1)) + sum(sqnorm, Flux.params(dudt)) + sum(sqnorm, Flux.params(MLP2))
Loss(θ,x,y) = mean( [ Flux.mse( y[:,i], predict(θ,x[:,i]) ) for i in 1:size(x,2)] ) + weight_decay * penalty()

## Generate predictions for the test set
## Parameters
N,NN = size(X_train);
epochs = 100; 
early_stop = 20
mb_size = 10;
num_of_increase = 3

## Vars
train_loss = []
test_loss = []
Qtst = []

for tt in 1:1
    "#"^50 |> println

    ## Callback function
    function cb()
        ltrn = Loss(model,X_train,y_train).data
        ltst = Loss(model,X_test,y_test).data

        push!(train_loss,ltrn)
        push!(test_loss,ltst)
    end

    train_error = Loss(model,X_train,y_train)
    test_error = Loss(model,X_test,y_test)
    @printf("Iter: %3d || Train Error: %2.6f || Test Error: %2.6f\n",
     0, train_error, test_error)
    @time for e in 1:epochs
        ## Weights for MLP, neural ODE, and MLP from inputs to outputs
        V1 = deepcopy(Flux.data.(params(MLP1)))
        V2 = deepcopy(Flux.data.(params(dudt)))
        V3 = deepcopy(Flux.data.(params(MLP2)))

        ## Inner loop
        learning_rate = 1e-2
        for mb in partition(randperm(size(X_train,2)), mb_size)
            l = Loss(model,X_train[:,mb],y_train[:,mb])
            Flux.back!(l)
            Flux.Optimise._update_params!(ADAM(learning_rate), params(MLP1))
            Flux.Optimise._update_params!(ADAM(learning_rate), params(dudt))
            Flux.Optimise._update_params!(ADAM(learning_rate), params(MLP2))
        end

        ## Reptile update
        for (w, v) in zip(params(MLP1),V1)
            dv = Flux.Optimise.apply!(ADAM(learning_rate), v, (w.data-v)/mb_size)
            @. w.data = v + dv
        end
        for (w, v) in zip(params(dudt),V2)
            dv = Flux.Optimise.apply!(ADAM(learning_rate), v, (w.data-v)/mb_size)
            @. w.data = v + dv
        end
        for (w, v) in zip(params(MLP2),V3)
            dv = Flux.Optimise.apply!(ADAM(learning_rate), v, (w.data-v)/mb_size)
            @. w.data = v + dv
        end
        
        train_error = Loss(model,X_train,y_train)
        test_error = Loss(model,X_test,y_test)
        @printf("Iter: %3d || Train Error: %2.6f || Test Error: %2.6f\n",
         e, train_error, test_error)
        
        ## Report
        (e%1 == 0 || e==1) && cb()
        if e>early_stop
            if sum(((test_loss[end-early_stop+1:end] - test_loss[end-early_stop:end-1]).>0)) > early_stop/2+1
                break
            end
        end
    end
end

## generare predictions for the test set and calculate the performance
println(" "^50)
println("Generare predictions for the test set and calculate the performance")
corr_test = zeros(size(y_test,1))
y_test_pred = hcat([reshape(predict(model,X_test[:,i]), Nr) for i in 1:size(X_test, 2)]...)
for i = 1:size(y_test,1)
    corr_test[i] = Flux.Tracker.data(cor(y_test_pred[i,:], y_test[i,:]))
end

p = sortperm(corr_test, rev=true)
y_plot = corr_test[p]
compound_names_ordered = compound_names[p]
y_plot_valid = y_plot[compound_names_ordered.==compound_names_ordered]
compound_names_ordered_valid = compound_names_ordered[compound_names_ordered.==compound_names_ordered]

writedlm("./results/predicted_metabolomic_profiles.csv",y_test_pred'.data, ',')
writedlm("./results/metabolites_corr.csv",corr_test, ',')

numOfTopMetabolites = 100
println("="^50)
println("The mean Spearman C.C. for all metabolites")
println(mean(y_plot))

println("="^50)
println("The mean Spearman C.C. for all annotated metabolites")
println(mean(y_plot_valid))

println("="^50)
println("Top 100 Spearman C.C. for annotated metabolites")
println(sort(y_plot_valid)[end-numOfTopMetabolites])

println("="^50)
println("The number of annotated metabolites with Spearman C.C. > 0.5")
println(sum(y_plot_valid .> 0.5))









