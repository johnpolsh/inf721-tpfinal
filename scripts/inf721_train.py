import torch
from torch import nn
from sklearn.metrics import accuracy_score
from inf721_model import *

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(our_model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
def train_one_epoch(model, batch_i, loss_function, optim, sched=None):
    model.train()
    avg_loss = .0
    run_loss = .0
    predictions = []
    targets = []
    for i, (x_t, y_t) in enumerate(train_dataloader):
        i += 1
        x_t = x_t.to(device)
        y_t = y_t.to(device)

        optim.zero_grad()

        if torch.cuda.is_available():
            y_hat = nn.parallel.data_parallel(model, x_t)
        else:
            y_hat = model(x_t)

        loss = loss_function(y_hat, y_t)
        avg_loss += loss.item()
        run_loss += loss.item()

        _, predicted = torch.max(y_hat, 1)
        predictions.extend(predicted.cpu().numpy())
        targets.extend(y_t.cpu().numpy())

        if i % batch_i == 0:
            print(f"\tbatch {i} avg loss {run_loss / batch_i}")
            run_loss = .0

        loss.backward()
        optim.step()
    
    if sched:
        sched.step()

    return (avg_loss / i, accuracy_score(targets, predictions))

acc_history_train = []
loss_history_train = []
def train(n_epochs, model, model_save_dir, loss_function, optim, sched=None):
    best_loss = 1_000_000
    for epoch in range(n_epochs):
        print(f"========================================================\nEPOCH {epoch + 1}")
        avg_loss, accuracy = train_one_epoch(model, 5, loss_function, optim, sched)
        loss_history_train.append(avg_loss)
        acc_history_train.append(accuracy)
        print(f"avg train loss: {avg_loss}, train accuracy: {accuracy * 100:.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_dir)

    print(f"\nTraining finished")

# Train 50 epochs
train(50, our_model, "our_model.pth", loss_fn, optimizer, scheduler)
save_model_for_resume(our_model, optimizer, "last-run.pth")

plt.figure(figsize=(16,9))

plt.subplot(1, 2, 1)
plt.plot(acc_history_train)
plt.xlabel('epochs')
plt.ylabel('accurary')
plt.title('train accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss_history_train)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('train loss')
plt.show()
