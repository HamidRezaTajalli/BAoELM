from elm_versions import elm, pca_transformed, pca_initialization, pruned_elm, drop_elm
from elm_versions import DRELM_main, TELM_Main, ML_ELM_main
from dataset_handler import mnist

def trainer(elm_type: str, dataset: str, trigger: str, hdlyr_size: int) -> None:



    elm_dict = {'poelm': elm.ELMClassifier, 'elm-pca': pca_transformed.PCTClassifier,
                'pca-elm': pca_initialization.PCIClassifier, 'pruned-elm': pruned_elm.PrunedClassifier,
                'drop-elm': drop_elm.DropClassifier}


    if elm_type.lower() == 'poelm':
        poelm = elm.ELMClassifier(hidden_layer_size=hdlyr_size, activation='sigm')
        poelm.fit(x_train, y_train)
        out = elm.predict(X)
    elif elm_type.lower() == 'elm-pca':
        pct = pca_transformed.PCTClassifier(hidden_layer_size=hdlyr_size, retained=None,
                               activation='sigm')  # retained can be (0, 1) percent variation or an integer number of PCA modes to retain
        pct.fit(X, y)
        res = pct.predict(X)


    elif elm_type.lower() == 'pca-elm':
        pci = el.PCIClassifier(retained=None,
                               activation='sigm')  # retained can be (0, 1) percent variation or an integer number of PCA modes to retain
        pci.fit(X, y)
        res = pci.predict(X)

    elif elm_type.lower() == 'pruned-elm':
        prune = el.PrunedClassifier(hidden_layer_size=hdlyr_size, activation='sigm')
        prune.fit(X, y)
        res = prune.predict(X)
    elif elm_type.lower() == 'drop-elm':
        drop = el.DropClassifier(hidden_layer_size=hdlyr_size, activation='sigm', dropconnect_pr=0.5, dropout_pr=0.5,
                                 dropconnect_bias_pctl=0.9, dropout_bias_pctl=0.9)
        drop.fit(X, y)
        res = drop.predict(X)

    elif elm_type.lower() == 'drelm':
        DRELM_main.DRELM_main()

    elif elm_type.lower() == 'telm':
        TELM_Main.TELM_main()

    elif elm_type.lower() == 'mlelm':
        ML_ELM_main.main_ML_ELM()





