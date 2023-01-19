import torch
import numpy as np
import matplotlib.pyplot as plt

def obj_func(Xtrain):
    y = torch.zeros(1).to(Xtrain.device)
    out = (-20*torch.exp(-0.2*torch.sqrt(0.5*(Xtrain**2 + y**2))) 
           - torch.exp(0.5*(torch.cos(2*torch.pi*Xtrain) + torch.cos(2*torch.pi*y)))
           + torch.e + 20).reshape(Xtrain.shape[0])
    return out

def gen_data(X_range):
    dim_list = []
    for dim_range in X_range.values():
        dim_list.append(np.arange(dim_range[0], dim_range[1], dim_range[2]))
    dim_list = eval(f"np.meshgrid({','.join(['dim_list[' + str(i) + ']' for i in range(len(dim_list))])})")
    Xtrain = torch.from_numpy(np.array(tuple([item.ravel() for item in dim_list])).T)
    ytrain = obj_func(Xtrain).reshape(Xtrain.shape[0], 1)
    return Xtrain, ytrain

def gen_observations(X_range, N):
    n_dims = len(X_range)
    X_initial = np.zeros((N, n_dims))
    for dim in range(n_dims):
        X_initial[:, dim] = np.random.uniform(X_range[dim][0], X_range[dim][1], size=N)
    X_initial = torch.from_numpy(X_initial)
    y_initial = obj_func(X_initial).reshape(X_initial.shape[0])
    return X_initial, y_initial

def gen_X_set(X_range, N):
    n_dims = len(X_range)
    X_out = np.zeros((N, n_dims))
    for dim in range(n_dims):
        X_out[:, dim] = np.random.uniform(X_range[dim][0], X_range[dim][1], size=N)
    X_out = torch.from_numpy(X_out)
    X_out = X_out.cuda() if torch.cuda.is_available() else X_out
    return X_out  

def read_result(filename, budget, runs):
    result = np.zeros((runs, budget + 1))
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [line[:-2].split(",") for line in lines]
        for i in range(runs):
            result[i, :] = [float(item) for item in lines[i]]
    return result  

def gen_grid():
    X_grid = torch.tensor(np.arange(-4, 4, 0.1))
    y_grid = obj_func(X_grid).reshape(X_grid.shape[0], 1)
    return X_grid, y_grid

def plot_1d(X, y, h, X_range, N, budget, batch, file_name):
    X_grid, y_grid = gen_grid()
    ncols = 10
    nrows = int(budget/ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(10*ncols, 10*nrows))
    for i in range(budget):
        row = i//ncols
        col = i%ncols
        if i==0:
            ax[row][col].scatter(X[:N], y[:N], c=(y[:N] > h))
        else:
            last_iter_stop = N + (i-1)*batch
            ax[row][col].scatter(X[last_iter_stop:(last_iter_stop + batch)], y[last_iter_stop:(last_iter_stop + batch)], color="red")
            ax[row][col].scatter(X[:last_iter_stop], y[:last_iter_stop], c=(y[:last_iter_stop] > h))
    plt.xlim([X_range[0][0], X_range[0][1]])
    plt.ylim([0, 10])
    plt.savefig(f"figures/{file_name}.png")
    plt.close(fig)
