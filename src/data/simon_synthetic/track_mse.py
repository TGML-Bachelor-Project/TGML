## Simons model training. This redo's some of the previous steps
    elif training_type == 2:

        ### Setup model
        model = ConstantVelocityModel(n_points=num_nodes, beta=model_beta)
        # model = SimonConstantVelocityModel(n_points=num_nodes, init_beta=model_beta)
        print('Model initial node start positions\n', model.z0)
        model = model.to(device)  # Send model to torch   
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        metrics = {'train_loss': [], 'test_loss': [], 'beta_est': []}

        

        num_train_samples = int(len(dataset)*training_portion)
        training_data = dataset[0:num_train_samples]
        test_data = dataset[num_train_samples:]
        n_train = len(training_data)
        training_batches = np.array_split(training_data, 450)
        batch_size = len(training_batches[0])

        gt_model = ConstantVelocityModel(n_points=num_nodes, beta=true_beta)
        # gt_model = SimonConstantVelocityModel(n_points=num_nodes, init_beta=model_beta)
        gt_dict = gt_model.state_dict()
        gt_z = torch.from_numpy(z0)
        gt_v = torch.from_numpy(v0)
        gt_dict["z0"] = gt_z
        gt_dict["v0"] = gt_v
        gt_model.load_state_dict(gt_dict)
        gt_nt = gt_model.eval()

        track_nodes = [0,3]
        tn_train = training_batches[-1][-1][2]
        tn_test = test_data[-1][2]

        def getres(t0, tn, f_model, track_nodes):
            time = np.linspace(t0, tn)
            res=[]
            for ti in time:
                res.append(torch.exp(f_model.log_intensity_function(track_nodes[0], track_nodes[1], ti)))
            return torch.tensor(res)

        res_gt = [getres(0, tn_train, gt_model, track_nodes), getres(tn_train, tn_test, gt_model, track_nodes)]
        print("Res_gt:")
        print(res_gt)


        model.z0.requires_grad, model.v0.requires_grad, model.beta.requires_grad = True, True, True
        gym_standard = SimonTrainTestGym(dataset=dataset, 
                    model=model, 
                    device=device, 
                    batch_size=train_batch_size, 
                    training_portion=training_portion,
                    optimizer=optimizer, 
                    metrics=metrics, 
                    time_column_idx=time_col_index,
                    num_epochs=num_epochs)

        
        model, training_losses, test_losses, track_dict = gym_standard.batch_train_track_mse(res_gt=res_gt,
                                            track_nodes=track_nodes,
                                            n_train=n_train,
                                            train_batches=training_batches,
                                            test_data=test_data)

        plotres(num_epochs, training_losses, test_losses, "LL Loss")
        plotres(num_epochs, track_dict["mse_train_losses"], track_dict["mse_test_losses"], "MSE Loss")
        plotgrad(num_epochs, track_dict["bgrad"], track_dict["zgrad"], track_dict["vgrad"])

