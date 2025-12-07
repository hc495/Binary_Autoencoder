import torch
from .training_objective import self_regression_loss, entropy_of_binary_hidden, entropy_eval, covarience_of_binary_hidden, L1_norm
import wandb

def auto_encoder_trainer(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    batch_size,
    num_epochs,
    direct_loss_function = self_regression_loss,
    entropy_weight = 0.1,
    covarience_weight = 0.1,
    cuda = True,
    entropy_threshold = 0,
    entropy_start_epoch = 0,
):
    """
    Train the autoencoder model.
    """
    # Set the model to training mode
    model.train()
    train_loss_log = []
    val_loss_log = []
    train_entropy_log = []
    val_entropy_log = []
    train_covarience_log = []
    val_covarience_log = []

    # Create data loaders for training and validation datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch
            if cuda:
                inputs = inputs.cuda()
            inputs = inputs.to(model.parameters().__next__().dtype)
            outputs, hidden = model(inputs)
            entropy = entropy_of_binary_hidden(hidden, threshold=entropy_threshold)
            covarience = covarience_of_binary_hidden(hidden)
            if epoch >= entropy_start_epoch:
                current_entropy_weight = entropy_weight
            else:
                current_entropy_weight = 0
            if entropy == 0:
                entropy = torch.tensor(0.0, device=hidden.device)
            if covarience == 0:
                covarience = torch.tensor(0.0, device=hidden.device)
            direct_loss = direct_loss_function(inputs, outputs)
            loss = direct_loss + current_entropy_weight * entropy
            loss += covarience_weight * covarience
            loss.backward()
            optimizer.step()
            wandb.log({
                "step": len(train_loss_log),
                "train_loss": loss.item(),
                "train_entropy": entropy.item(),
                "train_covarience": covarience.item(),
            })
            train_loss_log.append(loss.item())
            train_entropy_log.append(entropy.item())
            train_covarience_log.append(covarience.item())


        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {direct_loss.item():.4f}, Entropy: {entropy.item():.4f}, Covarience: {covarience.item():.4f}")

        # Validation step
        with torch.no_grad():
            val_loss = 0.0
            entropy = 0.0
            covarience = 0.0
            for batch in val_loader:
                inputs = batch
                if cuda:
                    inputs = inputs.cuda()
                inputs = inputs.to(model.parameters().__next__().dtype)
                outputs, hidden = model(inputs)
                val_loss += direct_loss_function(inputs, outputs).item()
                entropy += entropy_eval(hidden).item()
                covarience += covarience_of_binary_hidden(hidden).item()
            
            val_loss /= len(val_loader)
            val_entropy = entropy / len(val_loader)
            val_covarience = covarience / len(val_loader)
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_entropy": val_entropy,
                "val_covarience": val_covarience,
            })
            val_loss_log.append(val_loss)
            val_entropy_log.append(val_entropy)
            val_covarience_log.append(val_covarience)
            print(f"Validation Loss: {val_loss:.4f}, Validation Entropy: {val_entropy:.4f}, Validation Covarience: {covarience:.4f}")
    print("Training complete.")
    return {
        "train_loss_log": train_loss_log,
        "val_loss_log": val_loss_log,
        "train_entropy_log": train_entropy_log,
        "val_entropy_log": val_entropy_log,
        "train_covarience_log": train_covarience_log,
        "val_covarience_log": val_covarience_log
    }


def classifier_trainer(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    batch_size,
    num_epochs,
    direct_loss_function = self_regression_loss,
    entropy_weight = 0.1,
    covarience_weight = 0.1,
    cuda = True,
    entropy_threshold = 0,
    entropy_start_epoch = 0,
):
    """
    Train the autoencoder model.
    """
    # Set the model to training mode
    model.train()
    train_loss_log = []
    val_loss_log = []
    train_entropy_log = []
    val_entropy_log = []
    train_covarience_log = []
    val_covarience_log = []
    train_accuracy_log = []
    val_accuracy_log = []

    # Create data loaders for training and validation datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch[0], batch[1]
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs = inputs.to(model.parameters().__next__().dtype)
            outputs, hidden = model(inputs)
            entropy = entropy_of_binary_hidden(hidden, threshold=entropy_threshold)
            covarience = covarience_of_binary_hidden(hidden)
            if epoch >= entropy_start_epoch:
                current_entropy_weight = entropy_weight
            else:
                current_entropy_weight = 0
            if entropy == 0:
                entropy = torch.tensor(0.0, device=hidden.device)
            if covarience == 0:
                covarience = torch.tensor(0.0, device=hidden.device)
            direct_loss = direct_loss_function(outputs, labels)
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            loss = direct_loss_function(outputs, labels) + current_entropy_weight * entropy
            loss += covarience_weight * covarience
            loss.backward()
            optimizer.step()
            train_loss_log.append(loss.item())
            train_accuracy_log.append(accuracy.item())
            train_entropy_log.append(entropy.item())
            train_covarience_log.append(covarience.item())


        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {direct_loss.item():.4f}, Entropy: {entropy.item():.4f}, Covarience: {covarience.item():.4f}, Accuracy: {accuracy.item():.4f}")

        # Validation step
        with torch.no_grad():
            val_loss = 0.0
            entropy = 0.0
            covarience = 0.0
            accuracy = 0.0
            for batch in val_loader:
                inputs, labels = batch[0], batch[1]
                if cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                inputs = inputs.to(model.parameters().__next__().dtype)
                outputs, hidden = model(inputs)
                val_loss += direct_loss_function(outputs, labels).item()
                entropy += entropy_eval(hidden).item()
                covarience += covarience_of_binary_hidden(hidden).item()
                accuracy += (outputs.argmax(dim=1) == labels).float().mean().item()
            
            val_loss /= len(val_loader)
            val_entropy = entropy / len(val_loader)
            val_covarience = covarience / len(val_loader)
            val_accuracy = accuracy / len(val_loader)
            val_loss_log.append(val_loss)
            val_entropy_log.append(val_entropy)
            val_covarience_log.append(val_covarience)
            val_accuracy_log.append(val_accuracy)
            print(f"Validation Loss: {val_loss:.4f}, Validation Entropy: {val_entropy:.4f}, Validation Covarience: {covarience:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print("Training complete.")
    return {
        "train_loss_log": train_loss_log,
        "val_loss_log": val_loss_log,
        "train_entropy_log": train_entropy_log,
        "val_entropy_log": val_entropy_log,
        "train_covarience_log": train_covarience_log,
        "val_covarience_log": val_covarience_log,
        "train_accuracy_log": train_accuracy_log,
        "val_accuracy_log": val_accuracy_log
    }


def SAE_trainer(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    batch_size,
    num_epochs,
    direct_loss_function = self_regression_loss,
    L1_weight = 0.1,
    L1_start_epoch = 0,
    cuda = True,
    train_target_dataset = None, # only used for TransCoders
    val_target_dataset = None,
):
    """
    Train the autoencoder model.
    """
    # Set the model to training mode
    model.train()
    train_loss_log = []
    val_loss_log = []
    train_L1_log = []
    val_L1_log = []

    # Create data loaders for training and validation datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True if train_target_dataset is None else False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if train_target_dataset is not None:
        if len(train_target_dataset) != len(train_dataset):
            raise ValueError("train_target_dataset must be the same length as train_dataset")
        train_target_loader = torch.utils.data.DataLoader(train_target_dataset, batch_size=batch_size, shuffle=False)
    if val_target_dataset is not None:
        if len(val_target_dataset) != len(val_dataset):
            raise ValueError("val_target_dataset must be the same length as val_dataset")
        val_target_loader = torch.utils.data.DataLoader(val_target_dataset, batch_size=batch_size, shuffle=False)
        
    # Training loop
    for epoch in range(num_epochs):
        train_target_loader_iter = iter(train_target_loader) if train_target_dataset is not None else None
        val_target_loader_iter = iter(val_target_loader) if val_target_dataset is not None else None
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch
            if cuda:
                inputs = inputs.cuda()
            inputs = inputs.to(model.parameters().__next__().dtype)
            outputs, hidden = model(inputs)
            calculated_L1_norm = L1_norm(hidden)
            if epoch >= L1_start_epoch:
                current_L1_weight = L1_weight
            else:
                current_L1_weight = 0
            if train_target_dataset is None:
                direct_loss = direct_loss_function(inputs, outputs)
            else:
                target_batch = next(train_target_loader_iter)
                if cuda:
                    target_batch = target_batch.cuda()
                target_batch = target_batch.to(model.parameters().__next__().dtype)
                direct_loss = direct_loss_function(target_batch, outputs)
            loss = direct_loss + current_L1_weight * calculated_L1_norm
            loss.backward()
            optimizer.step()
            wandb.log({
                "step": len(train_loss_log),
                "train_loss": loss.item(),
                "train_L1": calculated_L1_norm.item(),
            })
            train_loss_log.append(loss.item())
            train_L1_log.append(calculated_L1_norm.item())


        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {direct_loss.item():.4f}, Entropy: {calculated_L1_norm.item():.4f}")

        # Validation step
        with torch.no_grad():
            val_loss = 0.0
            calculated_L1_norm = 0.0
            for batch in val_loader:
                inputs = batch
                if cuda:
                    inputs = inputs.cuda()
                inputs = inputs.to(model.parameters().__next__().dtype)
                outputs, hidden = model(inputs)
                if val_target_dataset is not None:
                    target_batch = next(val_target_loader_iter)
                    if cuda:
                        target_batch = target_batch.cuda()
                    target_batch = target_batch.to(model.parameters().__next__().dtype)
                    val_loss += direct_loss_function(target_batch, outputs).item()
                else:
                    val_loss += direct_loss_function(inputs, outputs).item()
                calculated_L1_norm += L1_norm(hidden).item()
            
            val_loss /= len(val_loader)
            val_L1_norm = calculated_L1_norm / len(val_loader)
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_L1": val_L1_norm,
            })
            val_loss_log.append(val_loss)
            val_L1_log.append(val_L1_norm)
            print(f"Validation Loss: {val_loss:.4f}, Validation Entropy: {val_L1_norm:.4f}")
    print("Training complete.")
    return {
        "train_loss_log": train_loss_log,
        "val_loss_log": val_loss_log,
        "train_L1_log": train_L1_log,
        "val_L1_log": val_L1_log
    }