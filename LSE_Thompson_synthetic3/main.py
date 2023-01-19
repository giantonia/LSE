import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import gen_X_set, obj_func, plot_1d, gen_observations, read_result
from models import sample_from_GP, model_training, evaluate_lse_model

def batch_lse_sample_uniform(X_range, n_candidates, model, h, batch):
    X_sample, y_sample = sample_from_GP(X_range, n_candidates, model)
    lse_idx = torch.where(y_sample >= h)[0]
    if lse_idx.shape[0] > 0:
        selected_cell_idx = lse_idx[np.random.randint(lse_idx.shape[0], size=batch)]
    else:
        selected_cell_idx = np.random.randint(X_sample.shape[0], size=batch)
    X_sample_point = X_sample[selected_cell_idx, :]
    return X_sample_point

def batch_nearest(X_range, n_candidates, model, h, batch):
    X_sample, y_sample = sample_from_GP(X_range, n_candidates, model)   
    selected_idx = torch.abs(y_sample - h).argsort()[:batch]
    return X_sample[selected_idx, :].reshape((batch, X_sample.shape[1]))

def batch_straddle(X_range, n_candidates, model, h, batch):
    # select random points
    X_random = gen_X_set(X_range, n_candidates)
    preds = model(X_random.float())
    mean_preds, var_preds = preds.mean, preds.variance
    std_pred = torch.sqrt(var_preds)
    straddle = 1.96*std_pred - torch.abs(mean_preds - h)
    selected_idx = straddle.argsort()[-batch:]
    selected_point = X_random[selected_idx, :]
    return selected_point.reshape((batch, X_random.shape[1]))

def exp(Xtrain, ytrain, 
        Xtest, ytest, 
        X_range, sample="random", 
        h=2, N=100, budget=100, 
        data_name="", n_candidates=None, 
        batch=8, note="", lr=0.1, iter=1000, 
        plot_sampling=False):
    f1_lcb_all = np.zeros((budget + 1))
    f1_mean_all = np.zeros((budget + 1))
    n_dims = Xtrain.shape[1]
    # train initial GP model
    model, likelihood = model_training(Xtrain, ytrain, lr, iter)
    # optimize iteratively
    for iteration in tqdm(range(1, budget + 1)):
        # evaluate the model
        f1_lcb, f1_mean = evaluate_lse_model(model, likelihood, Xtest, ytest, h)
        f1_lcb_all[iteration - 1] = f1_lcb
        f1_mean_all[iteration - 1] = f1_mean
        # sample new point
        if sample=="random":
            X_sample_point = gen_X_set(X_range, batch)
        elif sample=="uniform":
            X_sample_point = batch_lse_sample_uniform(X_range, n_candidates, model, h, batch)
        elif sample=="nearest":
            X_sample_point = batch_nearest(X_range, n_candidates, model, h, batch)
        elif sample=="straddle":
            X_sample_point = batch_straddle(X_range, n_candidates, model, h, batch)
        # evaluate new point
        y_sample_point = obj_func(X_sample_point)
        # update data and posterior model
        Xtrain = torch.cat((Xtrain, X_sample_point), 0)
        ytrain = torch.cat((ytrain, y_sample_point), 0)
        model, likelihood = model_training(Xtrain, ytrain, lr, iter)
    f1_lcb, f1_mean = evaluate_lse_model(model, likelihood, Xtest, ytest, h)
    print("Final F1: ", sample, " LCB: ", f1_lcb, " Mean: ", f1_mean)
    file_name = f"{data_name}_{sample}_h{h}_N{N}_NC{n_candidates}_budget{budget}_batch{batch}_{note}"
    if plot_sampling:
        plot_1d(Xtrain.cpu().detach(), ytrain.cpu().detach(), h, X_range, N, budget, batch, file_name)
    return file_name, f1_lcb_all, f1_mean_all

def save_result(file_name, f1_lcb_all, f1_mean_all):
    with open(f"result/lcb_{file_name}.txt", "a+") as f:
        out = list(f1_lcb_all)
        out = [str(item) for item in out]
        out = ",".join(out) + "\n"
        f.write(out)
    with open(f"result/mean_{file_name}.txt", "a+") as f:
        out = list(f1_mean_all)
        out = [str(item) for item in out]
        out = ",".join(out) + "\n"
        f.write(out)

def plot_final_result(budget, runs, data_name, h, N, n_candidates, batch, note):
    for pred in ["lcb", "mean"]:
        random_result = read_result(f"result/{pred}_{data_name}_random_h{h}_N{N}_NC{n_candidates}_budget{budget}_batch{batch}_{note}.txt", budget, runs)
        uniform_result = read_result(f"result/{pred}_{data_name}_uniform_h{h}_N{N}_NC{n_candidates}_budget{budget}_batch{batch}_{note}.txt", budget, runs)
        nearest_result = read_result(f"result/{pred}_{data_name}_nearest_h{h}_N{N}_NC{n_candidates}_budget{budget}_batch{batch}_{note}.txt", budget, runs)
        straddle_result = read_result(f"result/{pred}_{data_name}_straddle_h{h}_N{N}_NC{n_candidates}_budget{budget}_batch{batch}_{note}.txt", budget, runs)
        random_mean = np.mean(random_result, axis=0)
        random_std = np.std(random_result, axis=0)
        uniform_mean = np.mean(uniform_result, axis=0)
        uniform_std = np.std(uniform_result, axis=0)
        nearest_mean = np.mean(nearest_result, axis=0)
        nearest_std = np.std(nearest_result)
        straddle_mean = np.mean(straddle_result, axis=0)
        straddle_std = np.std(straddle_result, axis=0)
        fig, ax = plt.subplots(figsize=(10, 10))
        iterations = range(1, budget + 2)
        ax.plot(iterations, random_mean, color="green", label="random")
        ax.fill_between(iterations, random_mean - random_std, random_mean + random_std, alpha=0.3, color="green")
        ax.plot(iterations, uniform_mean, color="red", label="uniform")
        ax.fill_between(iterations, uniform_mean - uniform_std, uniform_mean + uniform_std, alpha=0.3, color="red")
        ax.plot(iterations, nearest_mean, color="black", label="nearest")
        ax.fill_between(iterations, nearest_mean - nearest_std, nearest_mean + nearest_std, alpha=0.3, color="black")
        ax.plot(iterations, straddle_mean, color="blue", label="straddle")
        ax.fill_between(iterations, straddle_mean - straddle_std, straddle_mean + straddle_std, alpha=0.3, color="blue")
        ax.legend()
        plt.savefig(f"figures/final_{pred}_{data_name}_h{h}_N{N}_NC{n_candidates}_budget{budget}_batch{batch}_{note}.png")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100, help="number of initial data points")
    # parser.add_argument("--max_dim", type=float, default=10.0, help="maximum of dimension range")
    # parser.add_argument("--n_dim", type=int, default=2, help="number of input dimensions")
    parser.add_argument("--test_size", type=int, default=5000, help="size of test set")
    parser.add_argument("--h", type=float, default=2.0, help="threshold")
    parser.add_argument("--budget", type=int, default=200, help="number of new evaluations")
    parser.add_argument("--runs", type=int, default=10, help="number of experiment runs")
    parser.add_argument("--n_candidates", type=int, default=100, help="number of candidate points")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--data_name", type=str, default="syn2d", help="data name")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate for optimization")
    parser.add_argument("--iter", type=int, default=300, help="number of iterations for optimization")
    parser.add_argument("--note", type=str, default="", help="note")
    args = parser.parse_args()

    X_range = {0: [-4, 4, 0.1]}
    for run in range(args.runs):
        print("Run ", run + 1)
        Xtrain, ytrain = gen_observations(X_range, args.N)
        Xtrain = Xtrain.cuda() if torch.cuda.is_available() else Xtrain
        ytrain = ytrain.cuda() if torch.cuda.is_available() else ytrain
        Xtest, ytest = gen_observations(X_range, args.test_size)
        Xtest = Xtest.cuda() if torch.cuda.is_available() else Xtest
        ytest = ytest.cuda() if torch.cuda.is_available() else ytest
        positive_percentage = torch.sum(ytest > args.h).cpu().detach().numpy()*100/ytest.shape[0]
        print(f"{positive_percentage}% of test data has positive label")
        run_results = {}
        for sample in ["random", "uniform", "nearest", "straddle"]:
            plot_sampling = True if run==args.runs-1 else False
            sample_result = exp(Xtrain, ytrain, 
                                Xtest, ytest, 
                                X_range, sample, 
                                h=args.h, 
                                N=args.N, 
                                budget=args.budget, 
                                data_name=args.data_name, 
                                n_candidates=args.n_candidates, 
                                batch=args.batch,
                                note=args.note, 
                                lr=args.lr, 
                                iter=args.iter, 
                                plot_sampling=plot_sampling)
            run_results[sample] = sample_result
        for result in run_results.values():
            file_name, f1_lcb_all, f1_mean_all = result[0], result[1], result[2]
            save_result(file_name, f1_lcb_all, f1_mean_all)
    plot_final_result(args.budget, args.runs, 
                      args.data_name, args.h, 
                      args.N, args.n_candidates, 
                      args.batch, args.note)