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
    using StatsBase: corspearman
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

weight_decay = 1e-5
Nh = 128
println("The hyperparameters are")
println(weight_decay, " ", Nh)

N_input = size(X_train, 1)
N_output = size(y_train, 1)
MLP1 = Chain(Dense(N_input,Nh,tanh))
dudt = Chain(Dense(Nh,Nh,tanh),
            Dense(Nh,Nh,tanh))
MLP2 = Chain(Dense(Nh,N_output))

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
mb_size = ceil(Int, NN/32);
num_of_increase = 3

## Vars
train_loss = []
test_loss = []
test_SCC = []
Qtst = []

for tt in 1:1
    "#"^50 |> println

    ## Callback function
    function cb()
        ltrn = Loss(model,X_train,y_train).data
        ltst = Loss(model,X_test,y_test).data

        push!(train_loss,ltrn)
        push!(test_loss,ltst)

        y_test_pred = hcat([reshape(predict(model,X_test[:,i]), N_output) for i in 1:size(X_test, 2)]...)
        for i = 1:size(y_test,1)
            corr_test[i] = Flux.Tracker.data(corspearman(y_test_pred[i,:], y_test[i,:]))
        end
        scc_tst = mean(corr_test[findall(compound_names.==compound_names)])
        push!(test_SCC,scc_tst)
    end

    train_error = Loss(model,X_train,y_train)
    test_error = Loss(model,X_test,y_test)
    y_test_pred = hcat([reshape(predict(model,X_test[:,i]), N_output) for i in 1:size(X_test, 2)]...)
    corr_test = zeros(size(y_test,1))
    for i = 1:size(y_test,1)
        corr_test[i] = Flux.Tracker.data(corspearman(y_test_pred[i,:], y_test[i,:]))
    end
    scc_tst = mean(corr_test[findall(compound_names.==compound_names)])
    @printf("Iter: %3d || Train Error: %2.6f || Test SCC: %2.6f\n",
     0, train_error, scc_tst)
    @time for e in 1:epochs
        ## Weights for MLP, neural ODE, and MLP from inputs to outputs
        V1 = deepcopy(Flux.data.(params(MLP1)))
        V2 = deepcopy(Flux.data.(params(dudt)))
        V3 = deepcopy(Flux.data.(params(MLP2)))

        ## Inner loop
        learning_rate = 5e-3
        for mb in partition(randperm(size(X_train,2)), mb_size)
            l = Loss(model,X_train[:,mb],y_train[:,mb])
            Flux.back!(l)
            Flux.Optimise._update_params!(ADAM(learning_rate), params(MLP1))
            Flux.Optimise._update_params!(ADAM(learning_rate), params(dudt))
            Flux.Optimise._update_params!(ADAM(learning_rate), params(MLP2))
        end

        ## Reptile update
        for (w, v) in zip(params(MLP1),V1)
            dv = Flux.Optimise.apply!(ADAM(learning_rate), v, (w.data-v))
            @. w.data = v + dv
        end
        for (w, v) in zip(params(dudt),V2)
            dv = Flux.Optimise.apply!(ADAM(learning_rate), v, (w.data-v))
            @. w.data = v + dv
        end
        for (w, v) in zip(params(MLP2),V3)
            dv = Flux.Optimise.apply!(ADAM(learning_rate), v, (w.data-v))
            @. w.data = v + dv
        end
        
        train_error = Loss(model,X_train,y_train)
        test_error = Loss(model,X_test,y_test)
        y_test_pred = hcat([reshape(predict(model,X_test[:,i]), N_output) for i in 1:size(X_test, 2)]...)
        for i = 1:size(y_test,1)
            corr_test[i] = Flux.Tracker.data(corspearman(y_test_pred[i,:], y_test[i,:]))
        end
        scc_tst = mean(corr_test[findall(compound_names.==compound_names)])
        @printf("Iter: %3d || Train Error: %2.6f || Test SCC: %2.6f\n",
         e, train_error, scc_tst)
        
        ## Report
        (e%1 == 0 || e==1) && cb()
        if e>early_stop
            if sum(((test_SCC[end-early_stop+1:end] - test_SCC[end-early_stop:end-1]).<0)) > early_stop/2+2
                break
            end
        end
    end
end

## Generate predictions for the test set and calculate the performance
println(" "^50)
println("Generate predictions for the test set and calculate the performance")
corr_test = zeros(size(y_test,1))
y_test_pred = hcat([reshape(predict(model,X_test[:,i]), Nr) for i in 1:size(X_test, 2)]...)
for i = 1:size(y_test,1)
    corr_test[i] = Flux.Tracker.data(corspearman(y_test_pred[i,:], y_test[i,:]))
end

p = sortperm(corr_test, rev=true)
corr_test_ordered = corr_test[p]
compound_names_ordered = compound_names[p]
corr_test_ordered_valid = corr_test_ordered[compound_names_ordered.==compound_names_ordered]
compound_names_ordered_valid = compound_names_ordered[compound_names_ordered.==compound_names_ordered]

numOfTopMetabolites = 50
println("="^50)
println("The mean Spearman C.C. for all metabolites")
println(mean(corr_test_ordered))

println("="^50)
println("The mean Spearman C.C. for all annotated metabolites")
println(mean(corr_test_ordered_valid))

println("="^50)
println("Top 100 Spearman C.C. for annotated metabolites")
println(sort(corr_test_ordered_valid)[end-numOfTopMetabolites])

println("="^50)
println("The number of annotated metabolites with Spearman C.C. > 0.5")
println(sum(corr_test_ordered_valid .> 0.5))


## Inferring microbe-metabolite interactions using the susceptibility method on the training set
println(" "^50)
println("Inferring microbe-metabolite interactions using the susceptibility method on the training set")
corr_train = zeros(size(y_train,1))
y_train_pred = hcat([reshape(predict(model,X_train[:,i]), Nr) for i in 1:size(X_train, 2)]...)
for i = 1:size(y_train,1)
    corr_train[i] = Flux.Tracker.data(corspearman(y_test_pred[i,:], y_test[i,:]))
end

susceptibility_all = zeros((Nb, Nr))
for i_species_perturbed = 1:Nb
    X_train_perturbed = copy(X_train)
    X_train_perturbed[i_species_perturbed, :] .= mean(X_train[i_species_perturbed, :])

    corr_train_perturbed = zeros(size(y_train,1))
    y_train_pred_perturbed = hcat([reshape(predict(model,X_train_perturbed[:,i]), Nr) for i in 1:size(X_train_perturbed, 2)]...)

    delta_x = repeat((X_train_perturbed[i_species_perturbed, :] - X_train[i_species_perturbed, :])', outer=[Nr,1])
    delta_y = (y_train_pred_perturbed - y_train_pred)
    susceptibility_each_sample = delta_y ./ delta_x
    susceptibility = mean(susceptibility_each_sample, dims=2)
    susceptibility_all[i_species_perturbed,:] = susceptibility.data
end

writedlm("./results/predicted_metabolomic_profiles.csv",y_test_pred'.data, ',')
writedlm("./results/true_metabolomic_profiles.csv",y_test', ',')
writedlm("./results/metabolites_corr.csv",corr_test, ',')
writedlm("./results/susceptibility_all.csv", susceptibility_all, ',')
println("Susceptibilies are saved to susceptibility_all.")



