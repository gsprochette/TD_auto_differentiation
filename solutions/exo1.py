analytical_grad = x / torch.norm(x, p='fro')
print(
    "Error between grad_y_x and analytical gradient: {:.2E}".format(torch.norm(analytical_grad - grad_y_x))
)
