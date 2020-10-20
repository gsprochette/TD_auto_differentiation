def f(x):
    """Function to minimize x -> ||Ax - y||^2"""
    return torch.norm(A@x - y)**2

# random initialization
x = torch.randn(p, requires_grad=True)

err_logs = np.zeros(niter)
for itr in tqdm(range(niter), leave=False):
    # compute functional and its gradient with respect to x
    sq_err = f(x)
    grad_x = grad(sq_err, [x])[0]
    
    # log error for future plot
    err_logs[itr] = sq_err.detach().numpy()
    
    # take gradient descent step
    x = x - lr * grad_x

plt.figure()
plt.plot(err_logs)
plt.yscale("log")
plt.title(r"Evolution of $\| Ax - y \|_2^2$ during gradient descent in log scale.")
plt.show()
    