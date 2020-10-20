def f(x):
    """Function to minimize x -> ||Ax - y||^2"""
    return torch.norm(A@x - y)**2

x_copy = x.clone().detach().requires_grad_(True)  # make a copy of the initial state for comparing with other optimizer
# this line is a little bit scary as we need to explicitely tell pytorch that this new tensor requires gradient computation,
# but does not affect x's gradient computation: we just want the initial state.

err_logs = np.zeros(niter)
for itr in tqdm(range(niter), leave=False, desc="SGD"):
    # set all gradients to zero
    optimizer.zero_grad()
    
    # compute error and gradient
    loss = f(x)
    loss.backward()  # now x.grad contains the gradient of loss with respect to x
    
    # save error for plot
    err_logs[itr] = loss.detach().numpy()
    
    # take gradient step
    optimizer.step()

plt.figure()
plt.plot(err_logs, label="SGD(lr={})".format(lr))

# now do the same with an unknown more advanced optimizer
x = x_copy
optimizer_scd = torch.optim.SGD([x], lr_scd, momentum=.9)
err_logs_scd = np.zeros(niter)
for itr in tqdm(range(niter), leave=False, desc="Adam"):
    # set all gradients to zero
    optimizer_scd.zero_grad()
    
    # compute error and gradient
    loss = f(x)
    loss.backward()  # now x.grad contains the gradient of loss with respect to x
    
    # save error for plot
    err_logs_scd[itr] = loss.detach().numpy()
    
    # take gradient step
    optimizer_scd.step()

plt.plot(err_logs_scd, label="SGD(lr={}, momentum={})".format(lr, 0.9))
plt.yscale("log")
plt.legend()
plt.title(r"Evolution of $\| Ax - y \|_2^2$ during gradient descent in log scale with SGD and Adam descent strategies.")
