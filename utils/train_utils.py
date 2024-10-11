import torch
import os
import copy

from model.model import temporal_loss, graph_loss

def train_pretrain(model, train_loader, val_loader, settings, use_val=True, vali_label_pos=2):
    model = model.to(device=settings["device"])
    losses = []
    graph_losses = []
    temporal_losses = []

    if use_val:
        validation_temporal_losess = []
        validation_graph_losess = []
        validation_losess = []
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr_rate"])

    model_save_folder = settings["model_save_folder"]
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    for epoch in range(settings["epochs"]):
        print("[epoch:{}/{}]".format(epoch + 1, settings["epochs"]))
        epoch_loss = []
        epoch_graph_loss = []
        epoch_temporal_loss = []
        i = 0
        for batch_features, batch_labels, batch_mask_features, _, graph, mask_graph, _ in train_loader:
            model.train()
            optimizer.zero_grad()

            batch_features = batch_features.squeeze().type(torch.float32).to(
                device=torch.device(settings["device"])
            )
            graph = graph.squeeze().type(torch.float32).to(
                device=torch.device(settings["device"])
            )
            mask_graph = mask_graph.squeeze().type(torch.float32).to(
                device=torch.device(settings["device"])
            )
            batch_mask_features = batch_mask_features.squeeze().type(torch.float32).to(
                device=torch.device(settings["device"])
            )

            X_output, _, graph_output = model(
                batch_mask_features, mask_graph, mode=1)
            validity_label = batch_labels.squeeze()[:, vali_label_pos].reshape(-1, 1, 1).type(torch.float32).to(
                device=torch.device(settings["device"])
            )


            temporal_reg_loss = temporal_loss(
                x_output=X_output, x_true=batch_features.detach(), mask=validity_label.detach()
            )

            mask = graph
            mask[mask != 0] = 1
            graph_mse_loss = graph_loss(
                graph=graph, graph_output=graph_output)
            loss = settings['graph_mse_loss_weight']*graph_mse_loss+temporal_reg_loss


            temporal_reg_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_temporal_loss.append(
                temporal_reg_loss.detach().cpu().numpy())
            epoch_graph_loss.append(graph_mse_loss.detach().cpu().numpy())

        losses.append(sum(epoch_loss)/len(epoch_loss))
        graph_losses.append(sum(epoch_graph_loss) / len(epoch_graph_loss))
        temporal_losses.append(
            sum(epoch_temporal_loss) / len(epoch_temporal_loss))


        if use_val:
            epoch_val_loss = []
            epoch_val_temporal_loss = []
            epoch_val_graph_loss = []
            with torch.no_grad():
                model.eval()
                for batch_features, batch_labels, batch_mask_features, _, graph, mask_graph, _ in val_loader:

                    batch_features = batch_features.squeeze().type(torch.float32).to(
                        device=torch.device(settings["device"])
                    )
                    graph = graph.squeeze().type(torch.float32).to(
                        device=torch.device(settings["device"])
                    )
                    batch_mask_features = batch_mask_features.squeeze().type(torch.float32).to(
                        device=torch.device(settings["device"])
                    )

                    X_output, X_Encoder_out, graph_output = model(
                        batch_mask_features, graph, mode=1)

                    validity_label = batch_labels.squeeze()[:, vali_label_pos].reshape(-1, 1, 1).type(torch.float32).to(
                        device=torch.device(settings["device"])
                    )

                    temporal_reg_loss = temporal_loss(
                        x_output=X_output, x_true=batch_features.detach(), mask=validity_label.detach()
                    )
                    mask = graph
                    mask[mask != 0] = 1
                    graph_mse_loss = graph_loss(
                        graph=graph, graph_output=graph_output)
                    loss = settings['graph_mse_loss_weight']*graph_mse_loss+temporal_reg_loss

                    epoch_val_loss.append(loss.detach().cpu().numpy())

                    epoch_val_temporal_loss.append(
                        temporal_reg_loss.detach().cpu().numpy()
                    )
                    epoch_val_graph_loss.append(
                        graph_mse_loss.detach().cpu().numpy())

                validation_losess.append(
                    sum(epoch_val_loss) / len(epoch_val_loss)
                )

                validation_temporal_losess.append(
                    sum(epoch_val_temporal_loss) / len(epoch_val_temporal_loss)
                )
                validation_graph_losess.append(
                    sum(epoch_val_graph_loss) / len(epoch_val_graph_loss)
                )
        print("epoch finish")

    try:
        torch.save(model, os.path.join(model_save_folder, "model.pt"))
    except:
        print("Saving model failed.")
    else:
        print("Saving model succeed.")

    return model

def train_predict(model, train_loader, val_loader, settings, use_val=True):
    model = model.to(device=settings["device"])
    loss_fn = torch.nn.MSELoss(reduction='none')

    training_losses = []

    if use_val:
        validation_losses = []


    best_loss = 1e9
    best_model = None

    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr_rate"])

    model_save_folder = settings["model_save_folder"]
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    for epoch in range(settings["epochs"]):
        print("[epoch:{}/{}]".format(epoch + 1, settings["epochs"]))
        epoch_training_loss = []
        for batch_features, batch_labels, _, _, graph, _, _ in train_loader:
            model.train()
            optimizer.zero_grad()

            batch_features = batch_features.squeeze().type(torch.float32).to(
                device=torch.device(settings["device"])
            )
            graph = graph.squeeze().type(torch.float32).to(
                device=torch.device(settings["device"])
            )
            batch_labels = (
                batch_labels.squeeze()[:, 0]
                .type(torch.float32)
                .to(device=settings["device"])
            )

            output = model(batch_features, graph).squeeze()

            # 计算loss
            loss = loss_fn(output, batch_labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_training_loss.append(loss.detach().cpu().numpy())

        training_losses.append(
            sum(epoch_training_loss) / len(epoch_training_loss))

        if use_val:
            epoch_val_loss = []
            with torch.no_grad():
                model.eval()
                for batch_features, batch_labels, _, _, graph, _, _ in val_loader:
                    batch_features = batch_features.squeeze().type(torch.float32).to(
                        device=torch.device(settings["device"])
                    )
                    graph = graph.squeeze().type(torch.float32).to(
                        device=torch.device(settings["device"])
                    )
                    batch_labels = (
                        batch_labels.squeeze()[:, 0]
                        .type(torch.float32)
                        .to(device=settings["device"])
                    )

                    output = model(batch_features, graph)

                    # 计算loss
                    loss = loss_fn(output, batch_labels)

                    epoch_val_loss.append(loss.detach().cpu().numpy())

                validation_losses.append(
                    sum(epoch_val_loss) / len(epoch_val_loss))

                if validation_losses[-1] < best_loss:
                    best_loss = validation_losses[-1]
                    best_model = copy.deepcopy(model)

        print("epoch finish")

    try:
        torch.save(
            best_model,
            os.path.join(model_save_folder, settings["best_model_name"]),
        )
    except:
        print("Saving best model failed.")
    else:
        print("Saving best model succeed.")

    try:
        torch.save(
            model,
            os.path.join(model_save_folder, settings["model_name"]),
        )
    except:
        print("Saving model failed.")
    else:
        print("Saving model succeed.")

