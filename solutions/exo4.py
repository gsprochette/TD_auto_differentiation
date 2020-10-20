def compute_loss(x, y):
    y_tilde = model(x)
    loss = F.binary_cross_entropy(y_tilde, y)
    return loss

def train_one_epoch(epoch_idx):
    for x_batch, y_batch in tqdm(dataloader, leave=False, desc=f"Training: Epoch {epoch_idx}"):
        # reset all gradients
        optimizer.zero_grad()
        
        # compute error
        loss = compute_loss(x_batch, y_batch)
        
        # compute gradients and take gradient step
        loss.backward()
        optimizer.step()

def compute_metrics(epoch_idx):
    with torch.no_grad():  # tell pytorch no gradient computation will be done: faster
        loss = 0
        accuracy = 0
        for x_batch, y_batch in tqdm(dataloader, leave=False, desc=f"Metrics: Epoch {epoch_idx}"):
            # loss: need to multiply by number of samples in the batch since the loss is an average
            loss += compute_loss(x_batch, y_batch).detach().numpy() * len(x_batch)
            
            # accuracy: count number of correct predictions
            prediction = model(x_batch) >= 0.5
            accuracy += torch.sum(prediction == y_batch).detach().numpy()
        
        # divide by number of samples
        loss = loss / len(dataset)
        accuracy = accuracy / len(dataset)
        
    return loss, accuracy
        

nepochs = 20  # numer of times the whole dataset is iterated upon
loss_logs, acc_logs = np.zeros(nepochs+1), np.zeros(nepochs+1)

# save initial loss and accuracy
loss, accuracy = compute_metrics(0)
tqdm.write("Epoch {}: loss {:.2E} - accuracy {:.1f}%".format(0, loss, 100*accuracy))
loss_logs[0] = loss
acc_logs[0] = accuracy

for epoch in tqdm(range(nepochs), leave=False, desc="Training"):
    # take gradient step on each sample
    train_one_epoch(epoch+1)
    
    # measure loss and accuracy on whole dataset
    loss, accuracy = compute_metrics(epoch+1)
    tqdm.write("Epoch {}: loss {:.2E} - accuracy {:.1f}%".format(epoch + 1, loss, 100*accuracy))
    
    # save loss and accuracy
    loss_logs[epoch+1] = loss
    acc_logs[epoch+1] = accuracy

    
# plotting
plt.figure(figsize=(12, 6))

# make two figures: one for the loss and one for the accuracy
plt.subplot2grid((1, 2), (0, 0))

# plot loss
plt.plot(loss_logs)
plt.yscale('log')
plt.title("Evolution of loss during gradient descent.")

# second figure
plt.subplot2grid((1, 2), (0, 1))

# plot accuracy
plt.axhline(y=0, color='k', ls='--')
plt.axhline(y=1, color='k', ls='--')
plt.plot(acc_logs)
plt.ylim(top=1.1, bottom=-.1)
plt.title("Evolution of accuracy during gradient descent.")

plt.show()
    