from elm_versions import elm, pca_transformed, pca_initialization, pruned_elm, drop_elm
from elm_versions import DRELM_main, TELM_Main, ML_ELM_main
from elm_versions import main_CNNELM, pseudoInverse
from dataset_handler import mnist, trigger
import torch


def trainer(elm_type: str, dataset: str, trigger_type: str, hdlyr_size: int) -> None:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # elm_dict = {'poelm': elm.ELMClassifier, 'elm-pca': pca_transformed.PCTClassifier,
    #             'pca-elm': pca_initialization.PCIClassifier, 'pruned-elm': pruned_elm.PrunedClassifier,
    #             'drop-elm': drop_elm.DropClassifier}
    ds_dict = {'mnist': mnist}

    trigger_obj = trigger.GenerateTrigger((4, 4), pos_label='upper-left', dataset=dataset, shape='square')
    all_data = ds_dict[dataset].get_alldata_simple()

    if elm_type.lower() == 'poelm':
        poelm = elm.ELMClassifier(hidden_layer_size=hdlyr_size)
        poelm.fit(all_data['train']['x'], all_data['train']['y_oh'])
        out = poelm.predict(all_data['test']['x'])
        acc = torch.sum(all_data['test']['y'] == torch.from_numpy(out)).item() / len(out)
        print(acc)

    elif elm_type.lower() == 'elm-pca':
        pct = pca_transformed.PCTClassifier(hidden_layer_size=hdlyr_size,
                                            retained=None)  # retained can be (0, 1) percent variation or an integer number of PCA modes to retain
        pct.fit(all_data['train']['x'], all_data['train']['y_oh'])
        out = pct.predict(all_data['test']['x'])
        acc = torch.sum(all_data['test']['y'] == torch.from_numpy(out)).item() / len(out)
        print(acc)

    elif elm_type.lower() == 'pca-elm':
        pci = pca_initialization.PCIClassifier(
            retained=None)  # retained can be (0, 1) percent variation or an integer number of PCA modes to retain
        pci.fit(all_data['train']['x'], all_data['train']['y_oh'])
        out = pci.predict(all_data['test']['x'])
        acc = torch.sum(all_data['test']['y'] == torch.from_numpy(out)).item() / len(out)
        print(acc)
    # TODO: Implement the rest of the elm types: dreelm, telm, ml-elm and also cnn-elm. what I should do: I should check the
    # TODO: the dataset handlers to fit each of these functions. I should also bring cnn-elm directory and put it down here. also add hidder-layer option. save path option with csv file which contains all settings and results
    # TODO: I have also check the availabitli of using cuda in each these modes. if the cuda can be used then use cuda else cpu.
    # TODO: don't forget to save the elapsed time!!
    # TODO: doon't forget to gc.collect!!
    elif elm_type.lower() == 'pruned-elm':
        prune = pruned_elm.PrunedClassifier(hidden_layer_size=hdlyr_size)
        prune.fit(all_data['train']['x'], all_data['train']['y_oh'])
        out = prune.predict(all_data['test']['x'])
        acc = torch.sum(all_data['test']['y'] == torch.from_numpy(out)).item() / len(out)
        print(acc)
    elif elm_type.lower() == 'drop-elm':
        drop = drop_elm.DropClassifier(hidden_layer_size=hdlyr_size, dropconnect_pr=0.3, dropout_pr=0.3,
                                       dropconnect_bias_pctl=None, dropout_bias_pctl=None)
        drop.fit(all_data['train']['x'], all_data['train']['y_oh'])
        out = drop.predict(all_data['test']['x'])
        acc = torch.sum(all_data['test']['y'] == torch.from_numpy(out)).item() / len(out)
        print(acc)

    elif elm_type.lower() == 'drelm':
        acc_train, acc_test, final_standard_div = DRELM_main.DRELM_main(all_data['train']['x'],
                                                                        all_data['train']['y_oh'].numpy(),
                                                                        all_data['test']['x'],
                                                                        all_data['test']['y_oh'].numpy())
        print(acc_test)

    elif elm_type.lower() == 'telm':
        acc_test, acc_train = TELM_Main.TELM_main(all_data['train']['x'],
                                            all_data['train']['y_oh'].numpy(),
                                            all_data['test']['x'],
                                            all_data['test']['y_oh'].numpy())
        print(acc_test)

    elif elm_type.lower() == 'mlelm':
        acc_test = ML_ELM_main.main_ML_ELM(all_data['train']['x'],
                                            all_data['train']['y_oh'].numpy(),
                                            all_data['test']['x'],
                                            all_data['test']['y_oh'].numpy())
        print(acc_test)

    elif elm_type.lower() == 'cnn-elm':
        dataloaders, classes_names = ds_dict[dataset].get_dataloaders_simple(batch_size=256, drop_last=False, is_shuffle=True)
        model = main_CNNELM.Net()
        model.to(device)
        optimizer = pseudoInverse.pseudoInverse(params=model.parameters(), C=1e-3)
        main_CNNELM.train(model, optimizer, dataloaders['train'])
        main_CNNELM.train_accuracy(model, dataloaders['train'])
        acc_test = main_CNNELM.test(model, dataloaders['test'])
        print(acc_test)
